Online Aggregation Algorithms with MongoDB
==========================================

Overview
--------

This module provides an implementation of **online Expectation-Maximization (EM) algorithms** for aggregating crowdsourced annotations.  
It supports storing and updating **task-class probabilities**, **worker confusion matrices**, and **class priors** directly in **MongoDB**.

Two main variants are provided (both are **abstract classes** and must be subclassed with concrete implementations of ``_e_step``, ``_m_step``, and ``_online_update_pi``):

- ``MongoOnlineAlgorithm``: variant using NumPy arrays (dense arrays)
- ``SparseMongoOnlineAlgorithm``: variant using ``sparse.COO`` backend for memory-efficient calculations

The module supports **online updates**, **batch processing**, and **per-batch EM convergence**.

.. note::

   For development, MongoDB is run in Docker.

Classes
-------

MongoOnlineAlgorithm
^^^^^^^^^^^^^^^^^^^^

Abstract base class for online EM aggregation with MongoDB.

**Features**

- MongoDB initialization and collection setup
- Insertion and retrieval of batch data
- Maintaining task, worker, and class mappings
- Online updates of:

  - ``T`` - task-class probabilities (shape ``n_tasks`` x ``n_classes``)
  - ``rho`` - class priors (shape ``n_classes``)
  - ``pi`` - worker confusion matrices (abstract; shape depends on implementation)
- Per-batch EM algorithm for convergence
- Optional ``top_k`` to keep only the most probable classes per task (not supported)

**Key Methods**

+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| **Method**                                                                                        | **Description**                                                     |
+===================================================================================================+=====================================================================+
| ``process_batch(batch, maxiter=50, epsilon=1e-6)``                                                | Processes a batch of votes, performs EM iterations until convergence |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``process_batch_matrix(batch_matrix, task_mapping, worker_mapping, class_mapping, maxiter, epsilon)`` | Core EM loop for a batch array.                                    |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``get_or_create_indices(collection, keys)``                                                       | Ensures mappings exist in MongoDB for tasks, workers, or classes.   |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``get_probas()``                                                                                  | Returns current task-class probability matrix ``T``.                |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``get_answers()``                                                                                 | Returns most probable class for each task.                          |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``get_answer(task_id)``                                                                           | Returns most probable class for a given task.                       |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``_process_batch_to_matrix(batch, ...)``                                                          | Converts batch to dense NumPy array.                                |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``_init_T(batch_matrix, ...)``                                                                    | Initializes task-class probability array.                           |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``_online_update_T(...)``                                                                         | Updates task-class probabilities online.                            |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``_online_update_rho(...)``                                                                       | Updates class priors online.                                        |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``_online_update_pi(...)``                                                                        | Updates worker confusion matrices (abstract).                       |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+
| ``_em_loop_on_batch(batch_matrix, ...)``                                                          | Internal EM loop per batch.                                         |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------+

**Properties**

+-------------+----------------------------------------------+
| **Property**| **Description**                              |
+=============+==============================================+
| ``T``       | Current global task-class probability matrix |
+-------------+----------------------------------------------+
| ``rho``     | Current global class priors                  |
+-------------+----------------------------------------------+
| ``gamma``   | Controls the learning rate                   |
+-------------+----------------------------------------------+
| ``n_classes`` | Global number of classes                   |
+-------------+----------------------------------------------+
| ``n_workers`` | Global number of workers                   |
+-------------+----------------------------------------------+
| ``n_task``  | Global number of tasks                       |
+-------------+----------------------------------------------+

**Abstract Method Expectations**

- ``_e_step(batch_matrix, batch_pi, batch_rho)``  
  Updates task-class probabilities using current batch estimates.  
  Must return:

  - ``batch_T`` : ``np.ndarray`` or ``sparse.COO`` — batch task-class probabilities  
  - ``denom_e_step`` : normalization array for likelihood computation  

- ``_m_step(batch_matrix, batch_T)``  
  Updates ``batch_rho`` (class priors) and ``batch_pi`` (worker confusion matrices).  
  Must return:

  - ``batch_rho`` : ``np.ndarray`` or ``sparse.COO``  
  - ``batch_pi`` : ``np.ndarray`` or ``sparse.COO``  

- ``_online_update_pi(worker_mapping, class_mapping, batch_pi)``  
  Responsible for updating global worker confusion matrices in MongoDB.

SparseMongoOnlineAlgorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^

Extends ``MongoOnlineAlgorithm`` to support **sparse matrices**.

**Features**

- Uses ``sparse.COO`` arrays to reduce memory usage.
- Overrides:

  - ``_process_batch_to_matrix``
  - ``_init_T``
  - ``_online_update_T``

- Supports optional ``top_k`` to keep only the most probable classes per task.

  - ``top_k``: optional integer to keep only the ``top_k`` most probable classes per task in MongoDB.
  - If set, probabilities for other classes are removed (``$unset``) from the MongoDB document.
  - Helps reduce storage and speeds up updates for large class sets.

MongoDB Schema
--------------

+---------------------------+-------------------------------------------------------------------------------------------+
| **Collection**            | **Structure**                                                                            |
+===========================+===========================================================================================+
| ``task_class_probs``      | ``{ "_id": task_id, "probs": { class_id: float }, "current_answer": class_id }``          |
+---------------------------+-------------------------------------------------------------------------------------------+
| ``class_priors``          | ``{ "_id": class_id, "prob": float }``                                                    |
+---------------------------+-------------------------------------------------------------------------------------------+
| ``worker_confusion_matrices`` | ``{ "_id": worker_id, "pi": { class_id: float } }`` *(abstract; implemented in subclasses)* |
+---------------------------+-------------------------------------------------------------------------------------------+
| ``task_mapping``          | ``{ "_id": task_id, "index": int }``                                                      |
+---------------------------+-------------------------------------------------------------------------------------------+
| ``worker_mapping``        | ``{ "_id": worker_id, "index": int }``                                                    |
+---------------------------+-------------------------------------------------------------------------------------------+
| ``class_mapping``         | ``{ "_id": class_id, "index": int }``                                                     |
+---------------------------+-------------------------------------------------------------------------------------------+
| ``user_votes``            | ``{ "_id": task_id, "votes": { user_id: vote } }``                                        |
+---------------------------+-------------------------------------------------------------------------------------------+

EM Algorithm Details
--------------------

``_em_loop_on_batch``
^^^^^^^^^^^^^^^^^^^^^

1. **Initialize ``batch_T`` matrix with ``_init_T``**

   **Purpose of ``_init_T``:**  
   Initializes the task-class probability matrix (``T``) for the current batch. It uses the batch matrix representation of task assignments:

   - Dense: ``np.ndarray`` of shape ``(n_tasks, n_users, n_classes)``
   - Sparse: ``sparse.COO`` of the same shape

   **Steps performed by ``_init_T``:**

   1. Computes initial probabilities by summing over workers for each task:

      .. code-block:: python

         T[task, class] = sum_over_workers(indicator(task_assigned_class))

   2. Normalizes probabilities for each task so they sum to 1.
   3. If previous probabilities exist in MongoDB (``task_class_probs``), combines them with the new batch using weighted averaging:

      .. math::

         T_{combined} = (1 - \gamma) * T_{existing} + \gamma * T_{batch}

   The result is an initialized ``batch_T`` matrix ready for EM iterations.

2. **Iteratively perform EM steps**

   - **``_m_step`` - Maximization step (abstract):**  
     Updates batch parameters ``batch_rho`` and ``batch_pi`` to maximize expected log-likelihood given current ``batch_T``.

   - **``_e_step`` - Expectation step (abstract):**  
     Updates task-class probabilities in ``batch_T`` based on current ``batch_rho`` and ``batch_pi``.

   - **Likelihood update and convergence check:**  
     Computes log-likelihood from ``denom_e_step`` and checks relative change against ``epsilon``.

3. **Termination criteria**

   Stop iterations when either:

   - Relative change in likelihood < ``epsilon``
   - Maximum iterations (``maxiter``) reached

4. **Online updates**

   Perform online updates on global variables: ``T``, ``rho``, and ``pi`` using batch estimates.

**Online Updates**

- ``T`` update: weighted average of old and new batch:

  .. math::

     T_{new} = (1 - \gamma) * T_{old} + \gamma * T_{batch}

- ``rho`` update: same weighted combination for class priors
- ``pi`` update: handled by subclass implementation
- ``\gamma = \gamma_0 / t^{decay}`` controls learning rate

Error Handling
--------------

- ``NotInitialized`` - raised if ``get_answers`` is called before initialization
- ``TaskNotFoundError`` - raised if querying a missing task
- ``DidNotConverge`` - warning if EM iterations did not converge
