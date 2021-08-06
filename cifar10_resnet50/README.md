# CIFAR10 Resnet50 Tutorial

**Adapted for Resnet50 and CIFAR10 from [Horovod's Fashion MNIST tutorial](https://github.com/horovod/tutorials/tree/master/fashion_mnist)**
 

In this tutorial, you will learn how to apply Horovod to a [ResNet50](https://arxiv.org/abs/1512.03385) model, trained on the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Prerequisites

If this is an in-person session, hosts will set up VM for you and provide you credentials to Jupyter Lab.  If you're working on this tutorial on your own, please follow installation instructions in [INSTALL.md](INSTALL.md).

Let's begin!

## Connect to Jupyter Lab

When you open Jupyter Lab in your browser, you will see a screen similar to this:

![image](https://user-images.githubusercontent.com/8098496/128559740-993036d0-c9a8-492c-a191-461265e5b702.png)

In this lab, we will use the Terminal and File Editor features.

## Setup Environment variables

## Explore model files

On the left hand side, you will see a number of Python files: `cifar10_resnet50.py`, `cifar10_resnet50_solution.py`, and a few intermediate files `cifar10_resnet50_after_step_N.py`.

<img src="https://user-images.githubusercontent.com/8098496/128559765-101650e7-6e16-4aba-8e94-272e48b57203.png" width="300"></img>

The first file contains the Keras model that does not have any Horovod code, while the second one has all the Horovod features added.  In this tutorial, we will guide you to transform `cifar10_resnet50.py` into `cifar10_resnet50_solution.py` step-by-step.  If you get stuck at any point, you can compare your code with the `cifar10_resnet50_after_step_N.py` file that corresponds to the step you're at.

Why Keras?  We chose Keras due to its simplicity, and the fact that it will be the way to define models in TensorFlow 2.0.

## Run cifar10_resnet50.py

Before we go into modifications required to scale our WideResNet model, let's run a single-GPU version of the model.

In the Launcher, click the Terminal button:

<img src="https://user-images.githubusercontent.com/16640218/53534695-d135d080-3ab4-11e9-830b-ea5a9e8581d1.png" width="300"></img>

In the terminal, type:

```
$ cp cifar10_resnet50.py cifar10_resnet50_backup.py
$ python cifar10_resnet50_backup.py --log-dir baseline
```

![image](https://user-images.githubusercontent.com/16640218/53534844-5620ea00-3ab5-11e9-9307-332db459da66.png)

After a few minutes, it will train a few epochs:

![image](https://user-images.githubusercontent.com/16640218/54184767-a4929900-4464-11e9-8a6a-e2fed3f4cd00.png)

Open the browser and load `http://<ip-address-of-vm>:6006/`:

![image](https://user-images.githubusercontent.com/16640218/54184664-69906580-4464-11e9-8a8f-3a0b4028b379.png)

You will see training curves in the TensorBoard.  Let it run.  We will get back to the results later.

## Modify cifar10_resnet50.py

Double-click `cifar10_resnet50.py` in the file picker, which will open it in the editor:

![image](https://user-images.githubusercontent.com/8098496/128559814-48054a5b-7f01-43f4-a5ec-a12b63ee0640.png)

Let's dive into the modifications!

### 1. Add Horovod import

Add the following code after `import tensorflow as tf`:

```python
import horovod.keras as hvd
```

![image](https://user-images.githubusercontent.com/8098496/128559846-2b856cf6-b54d-4714-bb76-55a04b96dcc7.png)
(see line 11)

### 2. Initialize Horovod

Add the following code after `args.checkpoint_format = os.path.join(args.log_dir, 'checkpoint-{epoch}.h5')`:

```python
# Horovod: initialize Horovod.
hvd.init()
```

![image](https://user-images.githubusercontent.com/8098496/128559904-05bf456e-eba4-492b-8016-bcb26d012bd2.png)
(see line 36-37)

### 3. Pin GPU to be used by each process

With Horovod, you typically use a single GPU per training process:

<img src="https://user-images.githubusercontent.com/16640218/53518255-7d5fc300-3a85-11e9-8bf3-5d0e8913c14f.png" width="400"></img>

This allows you to greatly simplify the model, since it does not have to deal with the manual placement of tensors.  Instead, you just specify which GPU you'd like to use in the beginning of your script.

Add the following code after `hvd.init()`:

```python
# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))
```

![image](https://user-images.githubusercontent.com/8098496/128559932-e68e872c-85e0-4ef6-ab20-7220260a7f42.png)
(see line 39-43)

### 4. Broadcast the starting epoch from the first worker to everyone else

In `cifar10_resnet50.py`, we're using the filename of the last checkpoint to determine the epoch to resume training from in case of a failure:

![image](https://user-images.githubusercontent.com/8098496/128559945-69f4006e-f88c-4eb7-9cbc-7e0ed0e280d0.png)

As you scale your workload to multi-node, some of your workers may not have access to the filesystem containing the checkpoint.  For that reason, we make the first worker to determine the epoch to restart from, and *broadcast* that information to the rest of the workers.

To broadcast the starting epoch from the first worker, add the following code:

```python
# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')
```

![image](https://user-images.githubusercontent.com/8098496/128559962-827c49e5-d2d5-4689-be3c-172308b70cea.png)
(see line 52-54)

### 5. Print verbose logs only on the first worker

Horovod uses MPI to run model training workers.  By default, MPI aggregates output from all workers.  To reduce clutter, we recommended that you write logs only on the first worker.

Replace `verbose = 1` with the following code:

```python
# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0
```

![image](https://user-images.githubusercontent.com/8098496/128559980-e8779004-0990-47a6-9c94-55bd5b268aae.png)
(see line 56-57)

### 6. Read checkpoint only on the first worker

For the same reason as above, we read the checkpoint only on the first worker and *broadcast* the initial state to other workers.

Replace the following code:

```python
# Restore from a previous checkpoint, if initial_epoch is specified.
if resume_from_epoch > 0:
    model = keras.models.load_model(args.checkpoint_format.format(epoch=resume_from_epoch))
else:
    ...
```

with:

```python
# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast both model and optimizer weights
# to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch))
else:
    ...
```

![image](https://user-images.githubusercontent.com/8098496/128559992-d0a3a3d9-b39c-4c87-94a1-a38a84a30197.png)
(see line 91-97)

### 7. Adjust learning rate and add Distributed Optimizer

Horovod uses an operation that averages gradients across workers.  Gradient averaging typically requires a corresponding increase in learning rate to make bigger steps in the direction of a higher-quality gradient.

Replace `opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)` with:

```python
# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.SGD(lr=args.base_lr * hvd.size(),
                           momentum=args.momentum)

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)
```

![image](https://user-images.githubusercontent.com/8098496/128560003-99ddbd77-93b9-4729-b056-5c943853a5c2.png)
(see line 114-119)

### 8. Add BroadcastGlobalVariablesCallback

In the previous section, we mentioned that the first worker would broadcast parameters to the rest of the workers.  We will use `horovod.keras.BroadcastGlobalVariablesCallback` to make this happen.

Add `BroadcastGlobalVariablesCallback` as the first element of the `callbacks` list:

```python
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    ...
```

![image](https://user-images.githubusercontent.com/8098496/128560076-032af016-4295-4764-a639-c89174aa1705.png)
(see line 137-140)

### 9. Add learning rate warmup

Many models are sensitive to using a large learning rate (LR) immediately after initialization and can benefit from learning rate warmup.  The idea is to start training with lower LR and gradually raise it to a target LR over a few epochs.  Horovod has the convenient `LearningRateWarmupCallback` for the Keras API that implements that logic.

Since we're already using `LearningRateScheduler` in this code, and it modifies learning rate along with `LearningRateWarmupCallback`, there is a possibility of a conflict.  In order to avoid such conflict, we will swap out `LearningRateScheduler` with Horovod `LearningRateScheduleCallback`.

Replace the following code:

```python
def lr_schedule(epoch):
    if epoch < 15:
        return args.base_lr
    if epoch < 25:
        return 1e-1 * args.base_lr
    if epoch < 35:
        return 1e-2 * args.base_lr
    return 1e-3 * args.base_lr


callbacks = [
    ...
    
    keras.callbacks.LearningRateScheduler(lr_schedule),
    ...
```

with:

```python
callbacks = [
    ...

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=verbose, initial_lr = args.base_lr * hvd.size()),

    # Horovod: after the warmup reduce learning rate by 10 on the 15th, 25th and 35th epochs.
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=args.warmup_epochs, end_epoch=15, multiplier=1., initial_lr = args.base_lr * hvd.size()),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=15, end_epoch=25, multiplier=1e-1, initial_lr = args.base_lr * hvd.size()),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=25, end_epoch=35, multiplier=1e-2, initial_lr = args.base_lr * hvd.size()),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=35, multiplier=1e-3, initial_lr = args.base_lr * hvd.size()),

    ...
```

![image](https://user-images.githubusercontent.com/8098496/128560096-2d01cd13-ab3a-4f0d-a8ba-b4d642917ff4.png)
(see line 144-156)

Since we've added a new `args.warmup_epochs` argument, we should register it:

```python
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
```

![image](https://user-images.githubusercontent.com/8098496/128560105-9ff14c1b-aaec-4795-86a4-cd2aae30545b.png)
(see line 26-27)

### 10. Save checkpoints & logs only of the first worker

We don't want multiple workers to be overwriting same checkpoint files, since it could lead to corruption.

Replace the following:

```python
callbacks = [
    ...

    keras.callbacks.ModelCheckpoint(args.checkpoint_format),
    keras.callbacks.TensorBoard(args.log_dir)
]
```

with:

```python
callbacks = [
    ...
]

# Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
    callbacks.append(keras.callbacks.TensorBoard(args.log_dir))
```

![image](https://user-images.githubusercontent.com/8098496/128560137-fd1e78b3-7491-42c9-8e11-1e6f8322aa5f.png)
(see line 162-164)

### 11. Modify training loop to execute fewer steps per epoch

To speed up training, we will execute fewer steps of distributed training.  To keep the total number of examples processed during the training the same, we will do `num_steps / N` steps, where `num_steps` is the original number of steps, and `N` is the total number of workers.

We will also speed up validation by validating `3 * num_validation_steps / N` steps on each worker.  The multiplier **3** provides over-sampling of validation data helps to increase probability that every validation example will be evaluated.

Replace `model.fit_generator(...)` with:

```python
# Train the model. The training will randomly sample 1 / N batches of training data and
# 3 / N batches of validation data on every worker, where N is the number of workers.
# Over-sampling of validation data, which helps to increase the probability that every
# validation example will be evaluated.
model.fit_generator(train_iter,
                    steps_per_epoch=len(train_iter) // hvd.size(),
                    callbacks=callbacks,
                    epochs=args.epochs,
                    verbose=verbose,
                    workers=4,
                    initial_epoch=resume_from_epoch,
                    validation_data=test_iter,
                    validation_steps=3 * len(test_iter) // hvd.size())
```

![image](https://user-images.githubusercontent.com/8098496/128560158-2a987f62-7faa-4790-a02c-98137467c16d.png)
(see line 166-178)

### 12. Average validation results among workers

Since we're not validating full dataset on each worker anymore, each worker will have different validation results.  To improve validation metric quality and reduce variance, we will average validation results among all workers.

To do so, inject `MetricAverageCallback` after `BroadcastGlobalVariablesCallback`:

```python
callbacks = [
    ...

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
    
    ...
```

![image](https://user-images.githubusercontent.com/8098496/128560179-3de1ebfd-72dc-491f-8fa2-877b31e630af.png)
(see line 144-148)

## Check your work

Congratulations!  If you made it this far, your `cifar10_resnet50.py` should now be fully distributed.  To verify, you can run the following command in the terminal, which should produce no output:

```
$ diff cifar10_resnet50.py cifar10_resnet50_solution.py
$
```

## Run distributed cifar10_resnet50.py

It's time to run your distributed `cifar10_resnet50.py`.  First, let's check if the single-GPU version completed.  Open the terminal, and verify that it did complete, and interrupt it using Ctrl-C if it did not.

![image](https://user-images.githubusercontent.com/16640218/53536718-448f1080-3abc-11e9-9e22-021dc3ba5de9.png)

Now, run distributed `cifar10_resnet50.py` using:

```
$ horovodrun -np 4 python cifar10_resnet50.py --log-dir distributed
```

![image](https://user-images.githubusercontent.com/16640218/53536888-da2aa000-3abc-11e9-9083-43060634433c.png)

After a few minutes, you should see training progress.  It will be faster compared to the single-GPU model:

![image](https://user-images.githubusercontent.com/16640218/53536956-270e7680-3abd-11e9-8f3b-acbe9bbfd085.png)

## Monitor training progress

Open the browser and load `http://<ip-address-of-vm>:6006/`:

![image](https://user-images.githubusercontent.com/16640218/54213792-3ec50200-44a2-11e9-9c7d-fdf9ab1bf94f.png)

By default, TensorBoard shows metric comparison based on the number of epochs.  This is shown on the chart above.  To compare training time it takes to achieve a certain accuracy, select **RELATIVE** in the *Horizontal Axis* selector:

![image](https://user-images.githubusercontent.com/16640218/54213965-94011380-44a2-11e9-9420-138bfe529ec6.png)

### Note

1. Since deep learning training is a stochastic process, you will see variation between accuracy of single-GPU and distributed training runs.  These are normal.

2. You will see approximately **3x** speedup in wall clock time, but not **4x** speedup.  This is expected for this model, since the model is very small and communication overhead plays large role in the training.  As you start training bigger models that take hours, days, or weeks to train, you will generally see better scaling efficiency.

## Parting thoughts

Thanks for following this tutorial!  We're excited to see you apply Horovod to speed up training of your models.
