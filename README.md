
# IR-GAIL
This is an example implementation of the paper "Cross Domain Robot Imitation with Invariant Representation". 

## Dependency

The experiments are dependent on ``gym``, ``mujoco-py``, and ``torch``. Make sure you have installed them properly.

## Get Started

### Train the agent with pretrained representation model
The ``example`` folder contains a quick demo on the InvertedDoublePendulum environment. 

To train a CDIL agent with pretrained representation network (in extrapolation mode), run the following command:

````
python ./example/idp.py --cuda --c1 1.3 --c2 1.5 --c3 1.4 --rollout_length 5000 --eval_interval 5000 --num_steps 5000000 --buffer ./assets/idp_expert_buffer.pth --embedding ./assets/idp_pretrained.pth --seed 0
````

In the above command, ``--c1 1.3 --c2 1.5 --c3 1.4`` specifies the environment parameter (1.3, 1.5, 1.4). The parameters of experts are around (0.9, 0.9, 0.9). You can change these parameters to do interpolation as well.

``--buffer ./assets/idp_expert_buffer.pth`` specifies the expert demonstration. 

``--embedding ./assets/idp_pretrained.pth`` specifies the pretrained representation network.

The return for this example will converge after 600k steps.

### Train the representation model

The random and expert experience used to train the representation network are in the ``./assets/idp_expert_random_buffer.pkl``.

If you want to train the representation network by yourself, run the following command:

````
python ./example/idp_train_representation.py
````

It will create a ``representation_logs`` folder, in which you can find the latest model as training goes on.

You can then use the trained model for imitation learning.

## Ablation
Navigate to the ``./example/idp.py``, and disable the dynamics loss by changing the ``c_f`` value to 0.0. Then, train the representation model and the agent again. 

This time, you will find the agent fail in the extrapolation experiment!

## Acknowledgement
We gratefully thank [ku2482][1] for a neat imitation learning framework. 🙂


[1]: https://github.com/ku2482/gail-airl-ppo.pytorch

