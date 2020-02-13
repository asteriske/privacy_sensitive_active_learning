from multiprocessing import Manager, Pool, Process, Queue
import numpy as np
from queue import Empty as Queue_Empty
import os
import time
import sys

from privacy_sensitive_active_learning.data import DataStore
from privacy_sensitive_active_learning import config
from privacy_sensitive_active_learning.db import sqlite_db
from privacy_sensitive_active_learning.model import JobSpec, NaiveModelProcess, OrdinalModelProcess
from privacy_sensitive_active_learning.util import init_logging

conf = config.load()
logger = init_logging(conf['multi']['log_level'], __name__)


def data_worker(write_q: Queue, read_q: Queue, work_q: Queue, data_store: DataStore):
    """
    Own all db read and write tasks.

    0. If data exists on the write queue, add it to the database.
    1a. If a request exists on the read queue retrieve the relevant data from the database.
    1b. Data retrieved from the database is wrapped in a JobSpec and added to the work queue
      for consumption by a fitter worker.
    """
    ttl = conf['multi']['data_worker_ttl']
    time_alive = ttl
    js_id = 0

    ppid = os.getppid()
    pid = os.getpid()
    
    logger.info("Data worker %s with parent %s", pid, ppid)

    while time_alive > 0:
        # Collecting data to insert

        try:
            data_item = write_q.get(block=False)
            if data_item['target'] != 'dn':
                logger.debug("Data item from write q: %s", data_item['ids'])
                logger.debug("Data item from write q type: %s", type(data_item['ids']))
            data_item['oracle_summary'] = data_store.draw_oracle_summary(data_item['ids'])


            data_store.add_sample(data_item)
            logger.debug("adding to db from write_q %s", data_item)
            write_queue_empty = False
            logger.debug("Length of write_q: %s", write_q.qsize())

        except Queue_Empty:
            
            write_queue_empty = True

        # Acting on data to output
        read_queue_empty = True

        try:

            data_req = read_q.get(block=False)

            logger.debug("drawing from db via read_q %s", data_req)

            X, y = data_store.provide_training_data(digit=data_req['digit'],
                n_values=None)
            logger.debug("drawn X for %s has dim %s", str(data_req['digit']), str(X.shape))
            
            logger.debug("Adding to work queue a job spec with ids size %s, features size %s, targeting %s, js_id %s", y.shape, X.shape, data_req['digit'], js_id)

            js = JobSpec(generation=data_req['generation'],
                         target=data_req['digit'],
                         ids=y,
                         features=X,
                         js_id=js_id
            )
            
            work_q.put(js)

            js_id+=1
            
            read_queue_empty = False

        except Queue_Empty:

            read_queue_empty = True

        if read_queue_empty and write_queue_empty:
            time_alive -= 1
            time.sleep(1)
            logger.debug("Data worker PID: %s, TTL: %s", ppid, time_alive)
            logger.debug("Work Queue Length: %s", work_q.qsize())
            # logger.debug("State of data_worker_stay_alive: %s", data_worker_stay_alive)

        else:
            time_alive = ttl

            data_store.db.commit()
            logger.debug("DB Commit.")
    logger.info("Data worker stopping.")
    data_store.db.close()


def fitter_worker(write_q: Queue, read_q: Queue, work_q: Queue, fit_type: str, ) -> None:
    """
    0. Draws a jobspec (containing data) from the work_q
    1. Fits a model on the data
    2. Predicts segmentation based upon the model
    3. Appends segmentation to the write_q to be inserted into the db
    4. If appropriate, appends another request onto the read_q
    """
    ttl = conf['multi']['fitter_worker_ttl']
    n_segments_reinsert = conf['model']['n_segments_reinsert']
    ppid = os.getppid()
    pid = os.getpid()
    logger.info("Fitter worker %s with parent %s", pid, ppid)

    
    time_alive = ttl
    while time_alive > 0:
        logger.info("Fitter worker TTL: %s", time_alive)
        logger.debug("Work Queue Length: %s", work_q.qsize())

        try:

            job_spec = work_q.get(block=False)
            logger.debug("Jobspec: generation %s, digit %s, data size %s", 
                job_spec.generation, job_spec.digit, str(job_spec.features.shape))

            ## Perform model fit

            logger.info("Fitting %s generation %s, js_id %s, process %s", job_spec.digit, job_spec.generation, job_spec.js_id, pid)

            if fit_type == "ordinal":
                mp = OrdinalModelProcess(job_spec)
            if fit_type == "naive":
                mp = NaiveModelProcess(job_spec)

            mp.fit()

            mp.save(os.path.join(f"models_{fit_type}", f"{job_spec.digit}.pkl"))

            ## Get segments out of model
            segments = mp.return_segment(X=X_base, n_samples=n_segments_reinsert)
 
            ## Put the segments in queue to be scored and written
            for s in segments:
                logger.debug("Segments to be added to write q: %s (pid: %s)", s, pid)
                
                if "," in s:
                    ids = [int(x) for x in s.split(',')]
                else:
                    ids = s

                write_q.put(
                    {'id_str': s,
                     'ids': ids,
                     'target': job_spec.digit
                     }
                )

            ## Queue up the next iteration to be built
            if job_spec.generation < conf['model']['model_iterations']:

                logger.debug("adding into read_q digit %s generation %s (pid %s)", job_spec.digit, (job_spec.generation+1), pid)

                read_q.put({
                    'digit': job_spec.digit,
                    'generation': (job_spec.generation + 1)
                })
            time_alive = ttl

        except Queue_Empty:
            time_alive -= 1
            time.sleep(1)
            continue

    logger.info("Fitter worker %s stopping", pid)


def multi(fit_type: str):
    """
    0. Initialize objects.
    1. Populate data queue with data to be added to DB.
    2. Populate work queue with one job for each digit.
    3. Initialize workers to consume jobs.
    """

    try:
        os.makedirs(f"models_{fit_type}")
    except:
        pass

    data_store = DataStore(fit_type)
    data_write_q = Queue()
    data_read_q = Queue()
    work_q = Queue()

    num_initial_draws = conf['model']['num_initial_draws']
    rows_per_draw = conf['model']['rows_per_draw']

    # The original dataset, made globally available for predictions
    global X_base
    X_base = data_store.trainX

    # Initial population of the data write queue
    ## populates segments into the db
    for _ in range(num_initial_draws):
        ids, _ = data_store.draw_random_sample(n=rows_per_draw)
        id_str = ",".join([str(x) for x in ids])

        data_write_q.put({
            'ids': ids,
            'id_str': id_str,
            'target': 'dn' # No target category
        })

    # Start the data worker to interact with the queues and manage the db

    data_proc = Process(target=data_worker, args=(data_write_q, data_read_q, work_q, data_store,))
    data_proc.start()

    time.sleep(2)

    # Initial population of the data read queue
    ## Creates requests for jobspecs to be created by reading from db

    for i in range(10):
        data_read_q.put({'generation':0,
                         'digit': f"d{i}"
        })

    time.sleep(20)
    
    workers = []
    for _ in range(conf['multi']['n_worker']):
        fitter_proc = Process(target=fitter_worker, args=(data_write_q, data_read_q, work_q, fit_type, ))
        workers.append(fitter_proc)
        fitter_proc.start()

    for proc in workers:
        proc.join()


    data_proc.join()
    
    logger.debug("All processes joined.")
    cur = sqlite_db(fit_type).cursor()
    cur.execute('select count(1) from samples')
    logger.debug("The database consists of %s rows", cur.fetchall())
