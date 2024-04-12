"""
2024-04-01
Tensorflow implementation of Automatic Domain Feature Extraction and Integration (ADFEI) framework.
The source code for the paper: Large-Scale Multi-Domain Recommendation: an Automatic Domain Feature Extraction and Integration Framework
@author: Anonymous
python = 3.6
tensorflow = 1.15.0
"""
import argparse
import multiprocessing
import os
import queue
import random
import shutil
import threading
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss
from collections import defaultdict


warnings.filterwarnings("ignore")


class GeneratorEnqueuer(object):
    """From keras source code training.py
    Builds a queue out of a data generator.

    # Arguments
        generator: a generator function which endlessly yields data
        pickle_safe: use multiprocessing if True, otherwise threading

        multiprocessing will go wrong, set pickle_safe to be False
    """

    def __init__(self, generator, pickle_safe=False):
        self._generator = generator
        self._pickle_safe = pickle_safe
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.finish = False

    def start(self, workers=1, max_q_size=10, wait_time=0.05):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
            wait_time: time to sleep in-between calls to put()
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._pickle_safe or self.queue.qsize() < max_q_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except StopIteration:
                    self.finish = True
                    break
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._pickle_safe:
                self.queue = multiprocessing.Queue(maxsize=max_q_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._pickle_safe:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed()
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except BaseException:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._pickle_safe:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._pickle_safe:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None


class ADFEI:
    def __init__(
        self,
        domains,
        use_domainid,
        file_folder,
        epoch,
        batch_size,
        embedding_dim,
        layers,
        keep_prob,
        batch_norm,
        lamda,
        lr,
        optimizer,
        verbose,
        activation,
        decay,
        early_stop,
        random_seed,
        gpu,
        prefix,
        label_index,
        feature_start_index,
        domain_id_index,
        user_id_index,
    ):
        """
        Init all parameters.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        assert (
            file_folder is not None
        ), "The file_folder is needed for getting all the data."
        enum_df = pd.read_csv(os.path.join(file_folder, "config.csv"))
        enum_cnt = dict(zip(enum_df.iloc[:, 0].tolist(), enum_df.iloc[:, 1].tolist()))

        train_chunks = pd.read_csv(
            os.path.join(file_folder, "{}_train1.csv".format(prefix)), chunksize=2
        )
        # get all feature columns
        for train_chunk in train_chunks:
            all_columns = train_chunk.columns.tolist()
            all_columns = [all_columns[domain_id_index]] + all_columns[
                feature_start_index:
            ]
            break

        columns_enum = {}
        for i in all_columns:
            if i in enum_cnt:
                columns_enum[i] = enum_cnt[i]
        self.columns_enum = columns_enum
        self.prefix = prefix
        self.user_id_index = user_id_index
        self.label_index = label_index
        self.feature_start_index = feature_start_index
        self.domain_id_index = domain_id_index
        self.domains = domains
        self.num_domains = len(domains)
        self.use_domainid = use_domainid
        self.epoch = epoch
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.keep_prob = keep_prob
        self.attention_dim = self.layers[-1]
        self.batch_norm = batch_norm
        self.lamda = lamda
        self.lr = lr
        self.optimizer = optimizer
        self.verbose = verbose
        self.activation = activation
        self.early_stop = early_stop
        self.file_folder = file_folder
        self.random_seed = random_seed
        self.all_columns = all_columns
        self.decay = decay
        self.version = int(time.time())
        self.best_epoch = 0
        self.trained = 0
        self.loaded = 0
        self.save_path = None

        assert len(self.layers) == len(
            self.keep_prob
        ), 'The length of "layers" and "keep_prob" should be same.'
        print(
            "Params:"
            + "\n epoch:{}".format(self.epoch)
            + "\n batch_size:{}".format(self.batch_size)
            + "\n embedding_dim:{}".format(self.embedding_dim)
            + "\n layers:{}".format(self.layers)
            + "\n keep_prob:{}".format(self.keep_prob)
            + "\n batch_norm:{}".format(self.batch_norm)
            + "\n lamda:{}".format(self.lamda)
            + "\n lr:{}".format(self.lr)
            + "\n optimizer:{}".format(self.optimizer)
            + "\n verbose:{}".format(self.verbose)
            + "\n activation:{}".format(self.activation)
            + "\n early_stop:{}".format(self.early_stop)
            + "\n file_folder:{}".format(self.file_folder)
            + "\n num_domains:{}".format(self.num_domains)
            + "\n use_domainid:{}".format(self.use_domainid)
            + "\n decay:{}".format(self.decay)
            + "\n random_seed:{}".format(self.random_seed)
        )

        self._init_graph()
        print("this model version is %d" % self.version)
        if not os.path.exists(os.path.join(self.file_folder, f"ADFEI_{self.version}")):
            os.makedirs(os.path.join(self.file_folder, f"ADFEI_{self.version}"))

    def _init_graph(self):
        """
        Init the tf graph for ADFEI model.
        """
        print("Init raw ADFEI graph")
        tf.set_random_seed(self.random_seed)
        tf.reset_default_graph()
        self.weights = self._initialize_weights()
        self.y = tf.placeholder(tf.float64, shape=[None, 1], name="y")
        self.inputs_placeholder = []
        feature_embedding = []

        if self.use_domainid == 0:
            self.inputs_placeholder.append(
                tf.placeholder(
                    tf.int64, shape=[None, 1], name=f"{self.all_columns[0]}_inp"
                )
            )
            tmp_channel_encode_inp = self.inputs_placeholder[-1]
            columns = self.all_columns[1:]
        else:
            columns = self.all_columns

        for column in columns:
            self.inputs_placeholder.append(
                tf.placeholder(tf.int64, shape=[None, 1], name=f"{column}_inp")
            )
            feature_embedding.append(
                tf.nn.embedding_lookup(
                    self.weights[f"{column}_embedding"], self.inputs_placeholder[-1]
                )
            )  # (N, 1, K) * D
        if self.use_domainid == 1:
            tmp_channel_encode_inp = self.inputs_placeholder[0]

        # hyper-parameter is_first_batch
        # hyper-parameter is_training
        self.inputs_placeholder.append(
            tf.placeholder(tf.int64, shape=[None, 1], name=f"is_first_batch_inp")
        )
        self.inputs_placeholder.append(
            tf.placeholder(tf.int64, shape=[None, 1], name=f"is_training_inp")
        )
        training_condition = self.inputs_placeholder[-1][0][0]
        firstbatch_condition = self.inputs_placeholder[-2][0][0]

        self.keep_prob = tf.cond(
            training_condition > 0,
            lambda: tf.constant(self.keep_prob),
            lambda: tf.constant(1.0, shape=[len(self.keep_prob)]),
        )
        self.bn_state = tf.cond(
            training_condition > 0,
            lambda: tf.constant(True),
            lambda: tf.constant(False),
        )

        domain_info = tf.cond(
            training_condition > 0,
            lambda: tf.one_hot(tmp_channel_encode_inp, depth=self.num_domains),
            lambda: tf.constant(1.0, shape=[1, 1, self.num_domains]),
        )

        feature_embedding = tf.squeeze(
            tf.keras.layers.concatenate(feature_embedding), [1]
        )  # (N, D*K)
        # shared
        shared_out = self.mlp_nn("shared", "shared_layer", feature_embedding)

        # unique
        unique_out = []
        domain_cur_mean = []
        domain_mean = []

        for domain_index in range(self.num_domains):
            unique_out.append(
                self.mlp_nn(f"domain{domain_index}", "unique_layer", feature_embedding)
            )
            domain_mean.append(
                tf.get_variable(
                    name=f"domain{domain_index}_mean",
                    shape=(1, self.layers[-1]),
                    initializer=tf.random_normal_initializer(),
                    trainable=False,
                )
            )
            # domain_info[0][0][domain_index]=1 if current sample is in domain[domain_index] else 0
            # if current sample is in domain[domain_index], then the out is mean(unique_out), i.e., the mean of all sample embedding
            # [1, K]
            domain_cur_mean.append(
                tf.cond(
                    training_condition > 0,
                    lambda: tf.reduce_mean(unique_out[-1], axis=0, keep_dims=True)
                    * domain_info[0][0][domain_index]
                    + (1 - domain_info[0][0][domain_index]) * domain_mean[-1],
                    lambda: domain_mean[-1],
                )
            )
            # history sample mean + current sample mean if not the firstbatch_condition
            # if not training_condition, domain_mean[-1] == domain_cur_mean[-1].
            domain_mean[-1] = tf.cond(
                firstbatch_condition * training_condition > 0,
                lambda: domain_cur_mean[-1],
                lambda: domain_mean[-1] * self.decay
                + domain_cur_mean[-1] * (1 - self.decay),
            )  # (1, K) * D

        self.out = []
        attention_out = self.attention_nn(
            "attention", domain_mean, feature_embedding, self.attention_dim
        )  # (N, K)
        for domain_index in range(self.num_domains):
            # unique_out[domain_index] : (N, K)
            self.out.append(
                self.out_nn(
                    f"domain{domain_index}_out",
                    shared_out,
                    unique_out[domain_index],
                    attention_out,
                    domain_mean[domain_index],
                )
            )

        # compute loss
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if self.lamda > 0:
            reg_loss = tf.add_n(reg_variables)
        else:
            reg_loss = 0

        # normal loss
        self.loss = [
            tf.losses.log_loss(labels=self.y, predictions=self.out[i]) + reg_loss
            for i in range(self.num_domains)
        ]

        shared_vars = [
            i
            for i in tf.trainable_variables()
            if "shared_layer" in i.name
            or "embedding" in i.name
            or "attention" in i.name
        ]
        unique_vars = [
            i
            for i in tf.trainable_variables()
            if "shared_layer" not in i.name
            and "embedding" not in i.name
            and "attention" not in i.name
        ]

        self.optimizers = []
        for domain_index in range(self.num_domains):
            if self.optimizer.lower() == "adam":
                cur_domain_optimizer_shared = tf.train.AdamOptimizer(
                    learning_rate=self.lr / self.num_domains
                ).minimize(self.loss[domain_index], var_list=shared_vars)
                cur_domain_optimizer_unique = tf.train.AdamOptimizer(
                    learning_rate=self.lr
                ).minimize(self.loss[domain_index], var_list=unique_vars)
            elif self.optimizer.lower() == "adagrad":
                cur_domain_optimizer_shared = tf.train.AdagradOptimizer(
                    learning_rate=self.lr / self.num_domains
                ).minimize(self.loss[domain_index], var_list=shared_vars)
                cur_domain_optimizer_unique = tf.train.AdagradOptimizer(
                    learning_rate=self.lr
                ).minimize(self.loss[domain_index], var_list=unique_vars)
            elif self.optimizer.lower() == "gd":
                cur_domain_optimizer_shared = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lr / self.num_domains
                ).minimize(self.loss[domain_index], var_list=shared_vars)
                cur_domain_optimizer_unique = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lr
                ).minimize(self.loss[domain_index], var_list=unique_vars)
            elif self.optimizer.lower() == "moment":
                cur_domain_optimizer_shared = tf.train.MomentumOptimizer(
                    learning_rate=self.lr / self.num_domains
                ).minimize(self.loss[domain_index], var_list=shared_vars)
                cur_domain_optimizer_unique = tf.train.MomentumOptimizer(
                    learning_rate=self.lr
                ).minimize(self.loss[domain_index], var_list=unique_vars)
            else:
                raise NotImplementedError("this optimizer not implemented...")
            cur_domain_optimizer = tf.group(
                cur_domain_optimizer_shared, cur_domain_optimizer_unique
            )
            self.optimizers.append(cur_domain_optimizer)

        # init
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession(
            config=tf.ConfigProto(gpu_options=gpu_options)
        )
        self.sess.run(init)
        print("ADFEI graph initialized...")

        # number of params
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        """
        Init weights.
        :return: all inited weights.
        """
        all_weights = dict()
        l2_reg = tf.contrib.layers.l2_regularizer(self.lamda)
        # embedding
        if self.use_domainid:
            columns = self.all_columns
        else:
            columns = self.all_columns[1:]
        for column in columns:
            all_weights[f"{column}_embedding"] = tf.get_variable(
                initializer=tf.random_normal(
                    shape=[self.columns_enum[column], self.embedding_dim],
                    mean=0.0,
                    stddev=0.01,
                ),
                regularizer=l2_reg,
                name=f"{column}_embedding",
            )

        return all_weights

    def mlp_nn(self, variable_scope, module_type, x):
        """
        The MLP network.
        :param variable_scope: variable_scope
        :param module_type: shared_layer or unique_layer
        :param x: input
        :return: mlp(x)
        """
        with tf.variable_scope(variable_scope):
            for i in range(len(self.layers)):
                cur_name = module_type + str(i + 1)
                cur_dim = self.layers[i]
                cur_dp = 1 - self.keep_prob[i]
                x = tf.keras.layers.Dense(
                    cur_dim, activation=self.activation, name=cur_name
                )(x)
                if self.batch_norm:
                    x = tf.keras.layers.BatchNormalization()(x, training=self.bn_state)
                x = tf.keras.layers.Dropout(cur_dp, name=cur_name + "_dp")(x)

            return x

    def out_nn(self, variable_scope, shared_out, unique_out, attention_out, domain_feature):
        """
        The output network.
        :param variable_scope: variable_scope
        :param shared_out: shared_out
        :param unique_out: unique_out
        :param attention_out: attention_out
        :param domain_feature: domain_feature
        :return: logits
        """
        with tf.variable_scope(variable_scope):
            x = tf.keras.layers.concatenate(
                [
                    shared_out,
                    unique_out,
                    attention_out,
                    domain_feature * tf.ones_like(unique_out),
                ]
            )
            x = tf.keras.layers.Dense(
                self.layers[-1], activation=self.activation, name="out_layer1"
            )(x)
            x = tf.keras.layers.Dense(1, activation=None, name="out_layer2")(x)
            # sigmoid
            x = tf.sigmoid(x, name=f"sigmoid_out")

            return x

    def attention_nn(self, variable_scope, means, query, atten_dim):
        """
        Attention network.
        :param variable_scope: variable_scope
        :param means: domain features list
        :param query: current sample feature
        :param atten_dim: attention dimension
        :return: attention output
        """
        # query: (N, K)
        # means: (1, K) * D
        with tf.variable_scope(variable_scope):
            # [D, K]
            inp = tf.keras.layers.concatenate(means, axis=0)
            # query # (N, K)
            query = tf.keras.layers.Dense(atten_dim, activation=None, name="key_layer")(
                query
            )
            # key (D, K)
            key = tf.keras.layers.Dense(atten_dim, activation=None, name="query_layer")(
                inp
            )
            # value (D, K)
            value = tf.keras.layers.Dense(
                atten_dim, activation=None, name="value_layer"
            )(inp)
            scores = tf.matmul(query, key, transpose_b=True) / np.sqrt(
                atten_dim
            )  # (N, D)
            norm_scores = tf.nn.softmax(scores, axis=1, name="softmax_layer")  # (N, D)
            atten = tf.matmul(norm_scores, value)  # (N, K)

            return atten

    def fit(self, pickle_safe=False, max_q_size=20, workers=1, wait_time=0.001):
        """
        fit the model.
        :param pickle_safe: whether is pickle_safe
        :param max_q_size: number of max_q_size
        :param workers: number of workers
        :param wait_time:  wait time
        :return:
        """
        train_paths = [
            os.path.join(self.file_folder, self.prefix + f"_train{i}.csv")
            for i in self.domains
        ]

        best_auc = -float("inf")
        es = 0
        self.best_epoch = 1

        for e in range(1, self.epoch + 1):
            print("============= epoch %s ==============" % e)
            try:
                train_gens = [
                    self.iterator(cur_train_path, shuffle=True)
                    for cur_train_path in train_paths
                ]
                train_enqueuers = [
                    GeneratorEnqueuer(train_gen, pickle_safe=pickle_safe)
                    for train_gen in train_gens
                ]

                for train_enqueuer in train_enqueuers:
                    train_enqueuer.start(max_q_size=max_q_size, workers=workers)
                n_batch = 1
                t1 = time.time()
                nb_samples = [0] * self.num_domains
                train_losses = [0] * self.num_domains
                while True:
                    train_generator_outs = []
                    flag = 0
                    for train_enqueuer in train_enqueuers:
                        while train_enqueuer.is_running():
                            if not train_enqueuer.queue.empty():
                                generator_output = train_enqueuer.queue.get()
                                train_generator_outs.append(generator_output)
                                flag = 1
                                break
                            elif train_enqueuer.finish:
                                train_generator_outs.append(None)

                                break
                            else:
                                time.sleep(wait_time)

                    if flag == 0:
                        break
                    for train_index in range(len(train_generator_outs)):
                        if train_generator_outs[train_index] is not None:
                            cur_x, cur_y = train_generator_outs[train_index]
                            cur_feedd = dict(zip(self.inputs_placeholder, cur_x))
                            cur_feedd.update({self.y: cur_y})
                            _, cur_loss = self.sess.run(
                                [self.optimizers[train_index], self.loss[train_index]],
                                feed_dict=cur_feedd,
                            )
                            train_losses[train_index] += cur_loss
                            nb_samples[train_index] += len(cur_y)
                            if self.verbose > 0:
                                if n_batch % self.verbose == 0:
                                    print(
                                        "[%d]Train loss of Domain-%d on step %d: %.6f"
                                        % (
                                            nb_samples[train_index],
                                            train_index,
                                            n_batch,
                                            train_losses[train_index] / n_batch,
                                        )
                                    )
                    n_batch += 1

                # valid
                t2 = time.time()
                cur_res = self.evaluate(training=True)

                self.print_info(
                    "Epoch %d [%.1f s]\t Dev" % (e, t2 - t1), cur_res, time.time() - t2
                )

                if self.early_stop > 0:
                    cur_auc = np.mean([v["AUC"] for k, v in cur_res.items()])
                    if cur_auc > best_auc:
                        self.best_epoch = e
                        best_auc = cur_auc
                        es = 0
                        self.save_path = self.saver.save(
                            self.sess,
                            save_path=os.path.join(
                                os.path.join(self.file_folder, f"ADFEI_{self.version}"),
                                f"ADFEI_{self.version}.ckpt",
                            ),
                            global_step=e,
                        )
                        print("model saved at %s" % self.save_path)
                    else:
                        es += 1
                        if es == self.early_stop:
                            print(
                                "Early stop at Epoch %d based on the best validation Epoch %d."
                                % (e, self.best_epoch)
                            )
                            break
                else:
                    self.save_path = self.saver.save(
                        self.sess,
                        save_path=os.path.join(
                            os.path.join(self.file_folder, f"ADFEI_{self.version}"),
                            f"ADFEI_{self.version}.ckpt",
                        ),
                        global_step=e,
                    )
                    self.best_epoch = e
                    print("model saved at %s" % self.save_path)
            finally:
                for train_enqueuer in train_enqueuers:
                    if train_enqueuer is not None:
                        train_enqueuer.stop()

        self.trained = 1

    def print_info(self, prefix, result, time):
        """
        print the information.
        :param prefix: print prefix
        :param result: result
        :param time: time
        :return:
        """
        line = [
            f"Domain-{i}: AUC:%.6f, GAUC:%.6f, logloss:%.6f"
            % (
                result["Domain_" + str(i)]["AUC"],
                result["Domain_" + str(i)]["GAUC"],
                result["Domain_" + str(i)]["logloss"],
            )
            for i in range(self.num_domains)
        ]
        print(prefix + ("[%.1f s]: \n" + "\n".join(line)) % time)

    def iterator(self, path, shuffle=False, test=0, use_user_id=False):
        """
        Generator of data.
        :param path: data path.
        :param shuffle: whether to shuffle the data. It should be True for training set.
        :param test: whether at the test/valid stage
        :param use_user_id: whether to return user_id
        :return: a batch data.
        """
        prefetch = 50  # prefetch number of batches.
        is_first_batch = 1
        batch_lines = []
        with open(path, "r") as fr:
            lines = []
            # remove csv header
            fr.readline()
            for prefetch_line in fr:
                lines.append(prefetch_line)
                if len(lines) >= self.batch_size * prefetch:
                    if shuffle:
                        random.shuffle(lines)
                    for line in lines:
                        batch_lines.append(line.strip().split(","))
                        if len(batch_lines) >= self.batch_size:
                            batch_array = np.array(batch_lines)
                            if test == 0 and is_first_batch == 1:
                                batch_array = np.concatenate(
                                    [
                                        batch_array,
                                        np.ones(shape=(self.batch_size, 1)),
                                        np.ones(shape=(self.batch_size, 1)),
                                    ],
                                    axis=1,
                                )
                                is_first_batch = 0
                            elif test == 1:
                                batch_array = np.concatenate(
                                    [
                                        batch_array,
                                        np.zeros(shape=(self.batch_size, 1)),
                                        np.zeros(shape=(self.batch_size, 1)),
                                    ],
                                    axis=1,
                                )
                            else:
                                batch_array = np.concatenate(
                                    [
                                        batch_array,
                                        np.zeros(shape=(self.batch_size, 1)),
                                        np.ones(shape=(self.batch_size, 1)),
                                    ],
                                    axis=1,
                                )
                            y = (
                                batch_array[:, self.label_index]
                                .astype("float64")
                                .reshape(-1, 1)
                            )
                            x = np.expand_dims(
                                batch_array[
                                    :,
                                    [self.domain_id_index]
                                    + list(
                                        range(
                                            self.feature_start_index,
                                            batch_array.shape[1],
                                        )
                                    ),
                                ]
                                .astype("float32")
                                .astype("int64")
                                .T,
                                2,
                            )
                            if use_user_id:
                                user_id = (
                                    batch_array[:, self.user_id_index]
                                    .astype("str")
                                    .reshape(-1, 1)
                                )
                                batch_lines = []
                                yield x, y, user_id
                            else:
                                batch_lines = []
                                yield x, y
                    lines = []
            if 0 < len(lines) < self.batch_size * prefetch:
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    batch_lines.append(line.split(","))
                    if len(batch_lines) >= self.batch_size:
                        batch_array = np.array(batch_lines)
                        if test == 0 and is_first_batch == 1:
                            batch_array = np.concatenate(
                                [
                                    batch_array,
                                    np.ones(shape=(self.batch_size, 1)),
                                    np.ones(shape=(self.batch_size, 1)),
                                ],
                                axis=1,
                            )
                            is_first_batch = 0
                        elif test == 1:
                            batch_array = np.concatenate(
                                [
                                    batch_array,
                                    np.zeros(shape=(self.batch_size, 1)),
                                    np.zeros(shape=(self.batch_size, 1)),
                                ],
                                axis=1,
                            )
                        else:
                            batch_array = np.concatenate(
                                [
                                    batch_array,
                                    np.zeros(shape=(self.batch_size, 1)),
                                    np.ones(shape=(self.batch_size, 1)),
                                ],
                                axis=1,
                            )
                        y = (
                            batch_array[:, self.label_index]
                            .astype("float64")
                            .reshape(-1, 1)
                        )
                        x = np.expand_dims(
                            batch_array[
                                :,
                                [self.domain_id_index]
                                + list(
                                    range(
                                        self.feature_start_index, batch_array.shape[1]
                                    )
                                ),
                            ]
                            .astype("float32")
                            .astype("int64")
                            .T,
                            2,
                        )
                        if use_user_id:
                            user_id = (
                                batch_array[:, self.user_id_index]
                                .astype("str")
                                .reshape(-1, 1)
                            )
                            batch_lines = []
                            yield x, y, user_id
                        else:
                            batch_lines = []
                            yield x, y
                if 0 < len(batch_lines) < self.batch_size:
                    batch_array = np.array(batch_lines)
                    if test == 0 and is_first_batch == 1:
                        batch_array = np.concatenate(
                            [
                                batch_array,
                                np.ones(shape=(batch_array.shape[0], 1)),
                                np.ones(shape=(batch_array.shape[0], 1)),
                            ],
                            axis=1,
                        )
                        is_first_batch = 0
                    elif test == 1:
                        batch_array = np.concatenate(
                            [
                                batch_array,
                                np.zeros(shape=(batch_array.shape[0], 1)),
                                np.zeros(shape=(batch_array.shape[0], 1)),
                            ],
                            axis=1,
                        )
                    else:
                        batch_array = np.concatenate(
                            [
                                batch_array,
                                np.zeros(shape=(batch_array.shape[0], 1)),
                                np.ones(shape=(batch_array.shape[0], 1)),
                            ],
                            axis=1,
                        )
                    y = (
                        batch_array[:, self.label_index]
                        .astype("float64")
                        .reshape(-1, 1)
                    )
                    x = np.expand_dims(
                        batch_array[
                            :,
                            [self.domain_id_index]
                            + list(
                                range(self.feature_start_index, batch_array.shape[1])
                            ),
                        ]
                        .astype("float32")
                        .astype("int64")
                        .T,
                        2,
                    )
                    if use_user_id:
                        user_id = (
                            batch_array[:, self.user_id_index]
                            .astype("str")
                            .reshape(-1, 1)
                        )
                        batch_lines = []
                        yield x, y, user_id
                    else:
                        batch_lines = []
                        yield x, y

    def save(self):
        """
        save as tf-serving
        """
        if not self.trained:
            print("Please first fit the model.")
            return
        # restore the best epoch model
        self.saver.restore(self.sess, save_path=self.save_path)

        # save as tf serving for online predict.
        self.serving_save_path = os.path.join(
            self.file_folder, f"ADFEI_{self.version}", "serving"
        )
        if os.path.exists(self.serving_save_path):
            shutil.rmtree(self.serving_save_path)

        builder = tf.saved_model.builder.SavedModelBuilder(self.serving_save_path)
        inputs = {}
        all_columns = self.all_columns + ["is_first_batch", "is_training"]
        for i in range(len(all_columns)):
            name = all_columns[i]
            inputs[name + "_inp:0"] = tf.saved_model.utils.build_tensor_info(
                self.inputs_placeholder[i]
            )
        outputs = {
            f"domain{i}_out:sigmoid_out:0": tf.saved_model.utils.build_tensor_info(
                self.out[i]
            )
            for i in range(self.num_domains)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs, outputs, tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(
            self.sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={"test_signature": signature},
            legacy_init_op=legacy_init_op,
        )
        builder.save()
        print("tf-serving model is saved in: {}".format(self.serving_save_path))

    def load(self, load_path=None, tf_serving=False):
        """
        load model.
        :param load_path:
        :param tf_serving: if load from tf_serving model
        """
        if not tf_serving:
            self.saver.restore(
                self.sess, load_path if load_path is not None else self.save_path
            )
        else:
            sess = tf.Session()
            meta_graph_def = tf.saved_model.loader.load(
                self.sess,
                [tf.saved_model.tag_constants.SERVING],
                load_path if load_path is not None else self.serving_save_path,
            )
            # get signature
            signature = meta_graph_def.signature_def
            key_my_signature = "test_signature"
            # get tensor name
            train_ids = []
            all_columns = self.all_columns + ["is_first_batch_inp", "is_training_inp"]
            for column in all_columns:
                train_id = (
                    signature[key_my_signature].inputs["{}_inp:0".format(column)].name
                )
                train_ids.append(sess.graph.get_tensor_by_name(train_id))
            preds = []
            for i in range(self.num_domains):
                preds.append(
                    sess.graph.get_tensor_by_name(
                        signature[key_my_signature]
                        .outputs[f"domain{i}_out:sigmoid_out:0"]
                        .name
                    )
                )
            self.loaded = True
            print(
                "Loaded tf-serving model from {}".format(
                    load_path if load_path is not None else self.serving_save_path
                )
            )

        self.loaded = 1

    def evaluate(self, valid=True, pickle_safe=False, max_q_size=20, workers=1, wait_time=0.001, training=False):
        """
        evaluate on valid or test dataset.
        :param valid: whether is valid or test mode.
        :param pickle_safe: pickle_safe
        :param max_q_size: number of max_q_size
        :param workers: number of workers
        :param wait_time: wait time
        :param training: whether is training mode
        :return: auc, gauc, loss
        """
        res = {}

        if not training and not self.trained and not self.loaded:
            print("Please first fit or load the model.")
            return

        try:
            if valid:
                valid_paths = [
                    os.path.join(self.file_folder, self.prefix + f"_val{i}.csv")
                    for i in self.domains
                ]
            else:
                valid_paths = [
                    os.path.join(self.file_folder, self.prefix + f"_test{i}.csv")
                    for i in self.domains
                ]

            valid_gens = [
                self.iterator(cur_valid_path, shuffle=True, test=1, use_user_id=True)
                for cur_valid_path in valid_paths
            ]
            valid_enqueuers = [
                GeneratorEnqueuer(valid_gen, pickle_safe=pickle_safe)
                for valid_gen in valid_gens
            ]

            for valid_index in range(len(valid_enqueuers)):
                valid_pred = []
                valid_label = []
                valid_user_id = []

                valid_enqueuer = valid_enqueuers[valid_index]
                valid_enqueuer.start(max_q_size=max_q_size, workers=workers)
                while True:
                    valid_generator_out = None
                    while valid_enqueuer.is_running():
                        if not valid_enqueuer.queue.empty():
                            valid_generator_out = valid_enqueuer.queue.get()
                            break
                        elif valid_enqueuer.finish:
                            break
                        else:
                            time.sleep(wait_time)
                    if valid_generator_out is None:
                        break

                    cur_x, cur_y, cur_user_id = valid_generator_out
                    cur_feedd = dict(zip(self.inputs_placeholder, cur_x))
                    logits = self.sess.run(self.out[valid_index], feed_dict=cur_feedd)

                    valid_pred.extend(logits.ravel())
                    valid_label.extend(cur_y.ravel())
                    valid_user_id.extend(cur_user_id.ravel())

                sigmoid_logits = np.array(valid_pred).astype("float64")
                labels = np.array(valid_label).astype("float64")
                user_ids = np.array(valid_user_id).astype("str")

                cur_auc = roc_auc_score(y_true=labels, y_score=sigmoid_logits)
                cur_logloss = log_loss(y_true=labels, y_pred=sigmoid_logits)
                cur_gauc = self.cal_group_auc(
                    labels=labels, preds=sigmoid_logits, user_id_list=user_ids
                )
                res["Domain_" + str(valid_index)] = {
                    "AUC": cur_auc,
                    "GAUC": cur_gauc,
                    "logloss": cur_logloss,
                }

        finally:
            for valid_enqueuer in valid_enqueuers:
                if valid_enqueuer is not None:
                    valid_enqueuer.stop()

        return res

    def cal_group_auc(self, labels, preds, user_id_list):
        """
        Calculate group auc (GAUC).
        :param labels: labels
        :param preds: preds
        :param user_id_list: user_id_list
        :return:
        """
        if len(user_id_list) != len(labels):
            raise ValueError(
                "impression id num should equal to the sample num,"
                "impression id num is {0}".format(len(user_id_list))
            )
        group_score = defaultdict(lambda: [])
        group_truth = defaultdict(lambda: [])
        for idx, truth in enumerate(labels):
            user_id = user_id_list[idx]
            score = preds[idx]
            truth = labels[idx]
            group_score[user_id].append(score)
            group_truth[user_id].append(truth)

        group_flag = defaultdict(lambda: False)
        for user_id in set(user_id_list):
            truths = group_truth[user_id]
            flag = False
            for i in range(len(truths) - 1):
                if truths[i] != truths[i + 1]:
                    flag = True
                    break
            group_flag[user_id] = flag

        impression_total = 0
        total_auc = 0
        #
        for user_id in group_flag:
            if group_flag[user_id]:
                auc = roc_auc_score(
                    np.asarray(group_truth[user_id]), np.asarray(group_score[user_id])
                )
                total_auc += auc * len(group_truth[user_id])
                impression_total += len(group_truth[user_id])
        group_auc = float(total_auc) / impression_total
        group_auc = round(group_auc, 4)
        return group_auc


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description="Run ADFEI.")
        parser.add_argument(
            "--domains", nargs="?", required=True, help="The domains to use."
        )
        parser.add_argument(
            "--use_domainid",
            type=int,
            required=True,
            help="whether to use domainId as a feature, 0 or 1.",
        )
        parser.add_argument("--epoch", type=int, default=10, help="Number of epochs.")
        parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
        parser.add_argument(
            "--embedding_dim", type=int, default=16, help="Number of embedding dim."
        )
        parser.add_argument(
            "--layers", nargs="?", default="[128,64,32]", help="Size of each MLP layer."
        )
        parser.add_argument(
            "--keep_prob",
            nargs="?",
            default="[0.9,0.8,0.7]",
            help="Keep probability. 1: no dropout.",
        )
        parser.add_argument(
            "--batch_norm",
            type=int,
            default=0,
            help="Whether to perform batch normaization (0 or 1)",
        )
        parser.add_argument(
            "--lamda",
            type=float,
            default=1e-6,
            help="Regularizer weight of embedding weight.",
        )
        parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
        parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            help="Specify an optimizer type (adam, adagrad, gd, moment).",
        )
        parser.add_argument(
            "--verbose",
            type=int,
            default=0,
            help="Whether to show the training process (0, or N batchs show one time)",
        )
        parser.add_argument(
            "--activation",
            type=str,
            default="relu",
            help="Which activation function to use for MLP layers: relu, sigmoid, tanh, identity",
        )
        parser.add_argument(
            "--decay", type=float, default=0.9, help="decay of moving average."
        )
        parser.add_argument(
            "--early_stop",
            type=int,
            default=1,
            help="Whether to perform early stop (0, 1 ... any positive integer)",
        )
        parser.add_argument("--gpu", type=str, default="0", help="Which gpu to use.")
        parser.add_argument(
            "--file_folder",
            type=str,
            required=True,
            help="folder saved training,validation,test data,config.csv and to save model.",
        )
        parser.add_argument(
            "--random_seed", type=int, default=2024, help="random seed."
        )
        parser.add_argument(
            "--prefix", type=str, default="data", help="prefix of train,valid,test data"
        )
        parser.add_argument(
            "--label_index", type=int, default=7, help="index of label in one sample"
        )
        parser.add_argument(
            "--feature_start_index",
            type=int,
            default=19,
            help="index of feature start in one sample",
        )
        parser.add_argument(
            "--domain_id_index",
            type=int,
            default=0,
            help="index of domain_id in one sample",
        )
        parser.add_argument(
            "--user_id_index",
            type=int,
            default=1,
            help="index of user_id in one sample",
        )
        return parser.parse_args()

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = ADFEI(
        domains=eval(args.domains),
        use_domainid=args.use_domainid,
        file_folder=args.file_folder,
        epoch=args.epoch,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        layers=eval(args.layers),
        keep_prob=eval(args.keep_prob),
        batch_norm=args.batch_norm,
        lamda=args.lamda,
        lr=args.lr,
        optimizer=args.optimizer,
        verbose=args.verbose,
        activation=args.activation,
        decay=args.decay,
        early_stop=args.early_stop,
        random_seed=args.random_seed,
        gpu=args.gpu,
        prefix=args.prefix,
        label_index=args.label_index,
        feature_start_index=args.feature_start_index,
        domain_id_index=args.domain_id_index,
        user_id_index=args.user_id_index,
    )

    model.fit()
    model.save()
    # load the early-stop model
    model.load()

    # test
    t2 = time.time()
    cur_res = model.evaluate(valid=False)
    model.print_info("Test", cur_res, time.time() - t2)
