# -*- coding: utf-8 -*-
import sys
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from fcgru import FCGRU

def format_to_3D (X, n_timestamps):
    '''
    input shape (n_samples,n_timestamps*n_bands) in order (t1b1,t1b2,..,t1bn,..,tmb1,tmb2,..,tmbn)
    output shape (n_samples,n_timestamps,n_bands)
    '''
    new_X = []
    for row in X :
        temp = np.split(row, n_timestamps, axis=0)
        new_X.append(temp)
    new_X = np.array(new_X)
    print (new_X.shape)
    return new_X

def format_label (y, n_classes, onehot=True) :
    '''
    output shape (n_samples,n_classes) if onehot is True otherwise (n_samples,1)
    '''
    encoder = LabelEncoder()
    y_tr = encoder.fit_transform(y)
    if onehot :
        y_tr = tf.keras.utils.to_categorical(y_tr, n_classes)
    print (y_tr.shape)
    return y_tr

def transform_label(test_label,test_prediction):
    '''
    Transform classification labels to input class values
    '''
    encoder = LabelEncoder()
    encoder.fit(test_label)
    prediction = encoder.inverse_transform(test_prediction)
    print (prediction.shape)
    return prediction

def get_batch(array, i, batch_size):
    '''
    Return a batch of input array
    '''
    start_id = i*batch_size
    end_id = min((i+1) * batch_size, array.shape[0])
    batch = array[start_id:end_id]
    return batch

def attention_mechanism(H,att_units,fcgru_units):
    '''
    Apply a customized attention mechanism on RNN outputs changing SoftMax in Tanh function
    '''
    W = tf.Variable(tf.random.normal([fcgru_units, att_units], stddev=0.1))
    b = tf.Variable(tf.random.normal([att_units], stddev=0.1))
    u = tf.Variable(tf.random.normal([att_units], stddev=0.1))
    
    v = tf.tanh(tf.tensordot(H, W, axes=1) + b)
    linear_lambdas = tf.tensordot(v, u, axes=1)
    linear_lambdas = tf.identity(linear_lambdas,name="att_scores")
    lambdas = tf.tanh(linear_lambdas,name="lambdas")

    output = tf.reduce_sum(H * tf.expand_dims(lambdas, -1), 1)
    output = tf.reshape(output,[-1,fcgru_units])
    return output

def rnn (X, fcgru_units, fc_units, n_timestamps, dropOut):
    '''
    Define the RNN model using the FCGRU cell
    '''
    X_seq = tf.unstack(X, axis=1)
    cell = FCGRU(fcgru_units,fc_units,dropOut)
    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-dropOut, state_keep_prob=1-dropOut)
    outputs,_ = tf.compat.v1.nn.static_rnn(cell, X_seq, dtype=tf.float32)
    outputs = tf.stack(outputs,axis=1)
    output = attention_mechanism(outputs, fcgru_units, fcgru_units)
    return outputs, output

def sensor_stream (X, fcgru_units, fc_units, n_timestamps, dropOut, scope_name):
    '''
    Create a branch for each source time series (radar/optical)
    '''
    with tf.compat.v1.variable_scope(scope_name):
        stream_hidden, stream_feat = rnn(X, fcgru_units, fc_units, n_timestamps, dropOut)
        stream_feat = tf.identity(stream_feat, name="learnt_features")
    return stream_hidden, stream_feat

def add_fc(features,units,n_classes,dropOut):
    '''
    Add fully connected layers to classify output features
    '''
    fc1 = tf.keras.layers.Dense(units,activation=None)(features)
    fc1 = tf.keras.layers.BatchNormalization(name="batchnorm1")(fc1)
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, rate=dropOut)

    fc2 = tf.keras.layers.Dense(units,activation=None)(fc1)
    fc2 = tf.keras.layers.BatchNormalization(name="batchnorm2")(fc2)
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, rate=dropOut)

    pred = tf.keras.layers.Dense(n_classes)(fc2)
    return pred

def initialize_uninitialized(sess):
    '''
    Function to initialize uninitialized variables when re-using 
    previous learned weights at the precedent level of hierarchy
    '''
    global_vars = tf.compat.v1.global_variables()
    is_not_initialized = sess.run([tf.compat.v1.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.compat.v1.variables_initializer(not_initialized_vars))

def run(train_X_rad, train_X_opt, train_y, valid_X_rad, valid_X_opt, valid_y, output_dir_models,
        split_numb, level, n_timestamps_rad, n_timestamps_opt, n_classes, fcgru_units, fc_units, 
        classif_units, batch_size, n_epochs, learning_rate, drop) :
    '''
    Define the computational graph
    '''
    n_bands_rad = train_X_rad.shape[-1]
    n_bands_opt = train_X_opt.shape[-1]

    # Placeholders
    X_rad = tf.compat.v1.placeholder(tf.float32,[None,n_timestamps_rad,n_bands_rad],name="X_rad")
    X_opt = tf.compat.v1.placeholder(tf.float32,[None,n_timestamps_opt,n_bands_opt],name="X_opt")
    if level is not None:
        y = tf.compat.v1.placeholder("float",[None,n_classes], name="y_level%s"%level)
    else:
        y = tf.compat.v1.placeholder("float",[None,n_classes], name="y")
    dropOut = tf.compat.v1.placeholder(tf.float32, shape=(), name="drop_rate")

    # Radar and Optical branches
    lst_feat = []

    stream_hidden_rad, stream_feat_rad = sensor_stream(X_rad,fcgru_units,fc_units,n_timestamps_rad,dropOut,"rad_stream")
    lst_feat.append(stream_feat_rad)

    stream_hidden_opt, stream_feat_opt = sensor_stream(X_opt,fcgru_units,fc_units,n_timestamps_opt,dropOut,"opt_stream")
    lst_feat.append(stream_feat_opt)

    # Features fusion with attention mechanism
    with tf.compat.v1.variable_scope("combined_feat"):
        hidden_feat = tf.concat([stream_hidden_rad,stream_hidden_opt],axis=1,name="hidden_features")
        combined_feat = attention_mechanism(hidden_feat,fcgru_units,fcgru_units)
        combined_feat = tf.identity(combined_feat,name="learnt_features")
    
    # Combining 3 feature sets (radar, optical, fused)
    weight = .5
    aux_pred = [] 

    if level is not None:
        pred_vs = "pred_level%s"%level
        cost_vs = "cost_level%s"%level
        optimizer_vs = "optimizer_level%s"%level
    else :
        pred_vs = "pred"
        cost_vs = "cost"
        optimizer_vs = "optimizer"

    with tf.compat.v1.variable_scope(pred_vs):
        for feat in lst_feat:
            aux_pred.append( tf.keras.layers.Dense(n_classes)(feat) )
        logits_full = add_fc(combined_feat,classif_units,n_classes,dropOut)
        score_tot = tf.nn.softmax(logits_full)

        for p in aux_pred:
            score_tot += weight * tf.nn.softmax(p) 
        prediction = tf.math.argmax(score_tot,1, name="prediction")
        correct = tf.math.equal(tf.math.argmax(score_tot,1),tf.math.argmax(y,1))
        accuracy = tf.reduce_mean(tf.dtypes.cast(correct,tf.float64))

    # Cost function
    with tf.compat.v1.variable_scope(cost_vs):
        cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logits_full))
        for p in aux_pred :
            cost += weight * tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=p))
    
    # Optimizer
    with tf.compat.v1.variable_scope(optimizer_vs):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Create a session and run the graph on training data
    n_batch = int(train_X_rad.shape[0]/batch_size)
    if train_X_rad.shape[0] % batch_size != 0:
        n_batch+=1
    print ("n_batch: %d" %n_batch)

    saver = tf.compat.v1.train.Saver()
    best_acc = sys.float_info.min

    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as session:
        session.run(init)

        for epoch in range(1,n_epochs+1):
            start = time.time()
            epoch_loss = 0
            epoch_acc = 0

            train_X_rad, train_X_opt, train_y = shuffle(train_X_rad, train_X_opt, train_y, random_state=0)

            for batch in range(n_batch):
                batch_X_rad = get_batch(train_X_rad,batch,batch_size)
                batch_X_opt = get_batch(train_X_opt,batch,batch_size)
                batch_y = get_batch(train_y,batch,batch_size)

                acc, loss, _ = session.run([accuracy, cost, optimizer], feed_dict={X_rad:batch_X_rad,
                                                                                   X_opt:batch_X_opt,
                                                                                   y:batch_y,
                                                                                   dropOut:drop})
                del batch_X_rad, batch_X_opt, batch_y

                epoch_loss += loss
                epoch_acc += acc
            
            stop = time.time()
            elapsed = stop - start
            print ("Epoch ",epoch, " Train loss:",epoch_loss/n_batch,"| Accuracy:",epoch_acc/n_batch, "| Time: ",elapsed)

            # At each epoch validate the model on validation set and save it if accuracy is better
            valid_batch = int(valid_X_rad.shape[0] / (4*batch_size))
            if valid_X_rad.shape[0] % (4*batch_size) != 0:
                valid_batch+=1

            total_pred = None
            for ibatch in range(valid_batch):
                valid_batch_X_rad = get_batch(valid_X_rad,ibatch,4*batch_size)
                valid_batch_X_opt = get_batch(valid_X_opt,ibatch,4*batch_size)

                batch_pred = session.run(prediction,feed_dict={X_rad:valid_batch_X_rad,
                                                               X_opt:valid_batch_X_opt,
                                                               dropOut:0.})
                del valid_batch_X_rad, valid_batch_X_opt

                if total_pred is None :
                    total_pred = batch_pred
                else : 
                    total_pred = np.hstack((total_pred,batch_pred))

            val_acc = accuracy_score(valid_y, total_pred)
            if val_acc > best_acc :
                print (np.bincount(np.array(total_pred)))
                print (np.bincount(np.array(valid_y)))

                print ("PREDICTION")
                print ("TEST F-Measure: %f" % f1_score(valid_y, total_pred, average='weighted'))
                print (f1_score(valid_y, total_pred, average=None))
                print ("TEST Accuracy: %f" % val_acc)
                if level is not None:
                    save_path = saver.save(session, output_dir_models+"/model_"+str(split_numb)+"_level-"+str(level))
                else:
                    save_path = saver.save(session, output_dir_models+"/model_"+str(split_numb))
                print ("Model saved in path: %s" % save_path)
                best_acc = val_acc

def restore_train (train_X_rad, train_X_opt, train_y, valid_X_rad, valid_X_opt, valid_y, output_dir_models,
                   split_numb, level, n_classes, classif_units,batch_size, n_epochs, learning_rate, drop):
    '''
    Restore previous learned variables and continue training on next level
    '''
    ckpt_path = os.path.join(output_dir_models,"model_%s_level-%s"%(str(split_numb),str(level-1)))
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session :
        model_saver = tf.compat.v1.train.import_meta_graph(ckpt_path+".meta")
        model_saver.restore(session, ckpt_path)
        
        graph = tf.compat.v1.get_default_graph()

        X_rad = graph.get_tensor_by_name("X_rad:0")
        X_opt = graph.get_tensor_by_name("X_opt:0")
        dropOut = graph.get_tensor_by_name("drop_rate:0")
        rad_feat = graph.get_tensor_by_name("rad_stream/learnt_features:0")
        opt_feat = graph.get_tensor_by_name("opt_stream/learnt_features:0")
        combined_feat = graph.get_tensor_by_name("combined_feat/learnt_features:0")
        print ("Model restored")
        
        y = tf.compat.v1.placeholder("float",[None,n_classes], name="y_level%s"%level)
        
        weight = .5
        aux_pred = []    

        with tf.compat.v1.variable_scope("pred_level%s"%level):
            aux_pred.append( tf.keras.layers.Dense(n_classes)(rad_feat) )
            aux_pred.append( tf.keras.layers.Dense(n_classes)(opt_feat) )

            logits_full = add_fc(combined_feat,classif_units,n_classes,dropOut)
            score_tot = tf.nn.softmax(logits_full)

            for p in aux_pred:
                score_tot += weight * tf.nn.softmax(p) 
            prediction = tf.math.argmax(score_tot,1, name="prediction")
            correct = tf.math.equal(tf.math.argmax(score_tot,1),tf.math.argmax(y,1))
            accuracy = tf.reduce_mean(tf.dtypes.cast(correct,tf.float64))

        with tf.compat.v1.variable_scope("cost_level%s"%level):
            cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logits_full))
            for p in aux_pred :
                cost += weight * tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=p))
                          
        with tf.compat.v1.variable_scope("optimizer_level%s"%level):
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize new variables and create new session for training
        initialize_uninitialized(session)

        n_batch = int(train_X_rad.shape[0]/batch_size)
        if train_X_rad.shape[0] % batch_size != 0:
            n_batch+=1
        print ("n_batch: %d" %n_batch)

        saver = tf.compat.v1.train.Saver()
        best_acc = sys.float_info.min

        for epoch in range(1,n_epochs+1):
            start = time.time()
            epoch_loss = 0
            epoch_acc = 0

            train_X_rad, train_X_opt, train_y = shuffle(train_X_rad, train_X_opt, train_y, random_state=0)

            for batch in range(n_batch):
                batch_X_rad = get_batch(train_X_rad,batch,batch_size)
                batch_X_opt = get_batch(train_X_opt,batch,batch_size)
                batch_y = get_batch(train_y,batch,batch_size)

                acc, loss, _ = session.run([accuracy, cost, optimizer], feed_dict={X_rad:batch_X_rad,
                                                                                   X_opt:batch_X_opt,
                                                                                   y:batch_y,
                                                                                   dropOut:drop})
                del batch_X_rad, batch_X_opt, batch_y

                epoch_loss += loss
                epoch_acc += acc
            
            stop = time.time()
            elapsed = stop - start
            print ("Epoch ",epoch, " Train loss:",epoch_loss/n_batch,"| Accuracy:",epoch_acc/n_batch, "| Time: ",elapsed)

            # Create a session for each epoch to validate model and save it if accuracy is better
            valid_batch = int(valid_X_rad.shape[0] / (4*batch_size))
            if valid_X_rad.shape[0] % (4*batch_size) != 0:
                valid_batch+=1

            total_pred = None
            for ibatch in range(valid_batch):
                valid_batch_X_rad = get_batch(valid_X_rad,ibatch,4*batch_size)
                valid_batch_X_opt = get_batch(valid_X_opt,ibatch,4*batch_size)

                batch_pred = session.run(prediction,feed_dict={X_rad:valid_batch_X_rad,
                                                               X_opt:valid_batch_X_opt,
                                                               dropOut:0.})
                del valid_batch_X_rad, valid_batch_X_opt

                if total_pred is None :
                    total_pred = batch_pred
                else : 
                    total_pred = np.hstack((total_pred,batch_pred))

            val_acc = accuracy_score(valid_y, total_pred)
            if val_acc > best_acc :
                print (np.bincount(np.array(total_pred)))
                print (np.bincount(np.array(valid_y)))

                print ("PREDICTION")
                print ("TEST F-Measure: %f" % f1_score(valid_y, total_pred, average='weighted'))
                print (f1_score(valid_y, total_pred, average=None))
                print ("TEST Accuracy: %f" % val_acc)
                save_path = saver.save(session, output_dir_models+"/model_"+str(split_numb)+"_level-"+str(level))
                print("Model saved in path: %s" % save_path)
                best_acc = val_acc

def restore_test (test_X_rad, test_X_opt, test_label, model_directory, split_numb, level, batch_size):
    '''
    Restore computational graph variables and run model on test set
    Save results in numpy array
    '''
    if level is not None:
        ckpt_path = os.path.join(output_dir_models,"model_%s_level-%s"%(str(split_numb),str(level)))
    else:
        ckpt_path = os.path.join(output_dir_models,"model_%s"%str(split_numb))

    results_path = os.path.join(model_directory,"results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session :
        model_saver = tf.compat.v1.train.import_meta_graph(ckpt_path+".meta")
        model_saver.restore(session, ckpt_path)
        
        graph = tf.compat.v1.get_default_graph()

        X_rad = graph.get_tensor_by_name("X_rad:0")
        X_opt = graph.get_tensor_by_name("X_opt:0")
        dropOut = graph.get_tensor_by_name("drop_rate:0")
        if level is not None:
            prediction = graph.get_tensor_by_name("pred_level%s/prediction:0"%level)
        else:
            prediction = graph.get_tensor_by_name("pred/prediction:0")

        print ("Model restored")

        n_batch = int(test_X_rad.shape[0] / (4*batch_size))
        if test_X_rad.shape[0] % (4*batch_size) != 0:
            n_batch+=1
        print ("n_batch: %d" %n_batch)

        total_pred = None

        for batch in range(n_batch):
            batch_X_rad = get_batch(test_X_rad,batch,(4*batch_size))
            batch_X_opt = get_batch(test_X_opt,batch,(4*batch_size))

            batch_pred = session.run(prediction,feed_dict={X_rad:batch_X_rad,X_opt:batch_X_opt,dropOut:0.})
            del batch_X_rad, batch_X_opt
            
            if total_pred is None :
                total_pred = batch_pred
            else : 
                total_pred = np.hstack((total_pred,batch_pred))
        
        total_pred = transform_label(test_label,total_pred)
        np.save(os.path.join(results_path,"results_"+str(split_numb)+".npy"),total_pred)

if __name__ == "__main__":

    # Reading data
    train_ts_rad = np.load(sys.argv[1])
    print ("train_ts_rad:",train_ts_rad.shape)
    train_ts_opt = np.load(sys.argv[2])
    print ("train_ts_opt:",train_ts_opt.shape)
    train_label = np.load(sys.argv[3])
    print ("train_label:", train_label.shape)
    
    valid_ts_rad = np.load(sys.argv[4])
    print ("valid_ts_rad:",valid_ts_rad.shape)
    valid_ts_opt = np.load(sys.argv[5])
    print ("valid_ts_opt:",valid_ts_opt.shape)
    valid_label = np.load(sys.argv[6])
    print ("valid_label:", valid_label.shape)

    test_ts_rad = np.load(sys.argv[7])
    print ("test_ts_rad:",test_ts_rad.shape)
    test_ts_opt = np.load(sys.argv[8])
    print ("test_ts_opt:",test_ts_opt.shape)
    test_label = np.load(sys.argv[9])
    print ("test_label:", test_label.shape)

    split_numb = int(sys.argv[10])
    output_dir_models = sys.argv[11]
    if not os.path.exists(output_dir_models):
        os.makedirs(output_dir_models)

    n_timestamps_rad = int(sys.argv[12])
    n_timestamps_opt = int(sys.argv[13])

    hier_pre = int(sys.argv[14])
    hier_pre_options = {1:True,2:False}
    print ("hier_pre:",hier_pre_options[hier_pre])
    
    sys.stdout.flush

    # Format data and label
    train_X_rad = format_to_3D(train_ts_rad, n_timestamps_rad)
    train_X_opt = format_to_3D(train_ts_opt, n_timestamps_opt)
    
    valid_X_rad = format_to_3D(valid_ts_rad, n_timestamps_rad)
    valid_X_opt = format_to_3D(valid_ts_opt, n_timestamps_opt)

    test_X_rad = format_to_3D(test_ts_rad, n_timestamps_rad)
    test_X_opt = format_to_3D(test_ts_opt, n_timestamps_opt)

    # Model Parameters
    fcgru_units = 512
    fc_units = 64
    classif_units = 512
    batch_size = 32
    n_epochs = 2000
    learning_rate = 1E-4
    drop = 0.4

    if hier_pre == 1: # Hierarchical classification strategy
        n_level = train_label.shape[1]

        for level in range(1,n_level):
            print ("level",level)
            train_y = train_label[:,level] 
            train_y = train_y.astype('int64')
            valid_y = valid_label[:,level]
            valid_y = valid_y.astype('int64')

            n_classes = len(np.unique(train_y))
            train_y = format_label(train_y,n_classes)
            valid_y = format_label(valid_y,n_classes,onehot=False)
            
            if level == 1 :
                run (train_X_rad, train_X_opt, train_y, valid_X_rad, valid_X_opt, valid_y, output_dir_models,
                    split_numb, level, n_timestamps_rad, n_timestamps_opt, n_classes, fcgru_units, fc_units, 
                    classif_units, batch_size, n_epochs, learning_rate, drop)
            else :
                restore_train(train_X_rad, train_X_opt, train_y, valid_X_rad, valid_X_opt, valid_y, output_dir_models,
                        split_numb, level, n_classes, classif_units,batch_size, n_epochs, learning_rate, drop)
        
        test_label = test_label[:,-1]
        test_label = test_label.astype('int64')
        restore_test(test_X_rad, test_X_opt, test_label, output_dir_models, split_numb, level, batch_size)

    elif hier_pre == 2 : # Simple classification
        train_y = train_label[:,-1] 
        train_y = train_y.astype('int64')
        valid_y = valid_label[:,-1]
        valid_y = valid_y.astype('int64')
        test_label = test_label[:,-1]
        test_label = test_label.astype('int64')

        n_classes = len(np.unique(train_y))
        train_y = format_label(train_y,n_classes)
        valid_y = format_label(valid_y,n_classes,onehot=False)

        run (train_X_rad, train_X_opt, train_y, valid_X_rad, valid_X_opt, valid_y, output_dir_models,
                    split_numb, None, n_timestamps_rad, n_timestamps_opt, n_classes, fcgru_units, fc_units, 
                    classif_units, batch_size, n_epochs, learning_rate, drop)
        
        restore_test(test_X_rad, test_X_opt, test_label, output_dir_models, split_numb, None, batch_size)