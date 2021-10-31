from gym.envs.registration import register
import os

def coeff_to_xml(coeff):
    return coeff[0], coeff[0], coeff[1], coeff[2] * 0.045, coeff[2] * 0.3, coeff[1]

def get_coeff_env_name(coeff):
    return 'InvertedDoublePendulum2_{}_{}_{}-v2'.format(coeff[0], coeff[1], coeff[2])

def get_coeff_file_name(coeff):
    return 'inverted_dp2_{}_{}_{}.xml'.format(coeff[0], coeff[1], coeff[2])

def dump_xml(coeff):
    with open(os.path.join(os.path.dirname(__file__), 'assets/form.xml'), 'r') as f:
        content = f.read()
        # print(content)
        content_replaced = content.format(*coeff_to_xml(coeff))
        # print(content_replaced)

        # input()
        with open(os.path.join(os.path.dirname(__file__), 'assets/' + get_coeff_file_name(coeff)), 'a+') as wf:
            wf.write(content_replaced)
            wf.flush()

def register_inverted_double_pendulum(coeff):
    kwargs = {
        'coeff': coeff,
    }

    dump_xml(coeff)

    register(
        id=get_coeff_env_name(coeff),
        entry_point='custom_env.idp2.inverted_double_pendulum:InvertedDoublePendulumEnv',
        max_episode_steps=1000,
        nondeterministic=True,
        kwargs=kwargs
    )

    return get_coeff_env_name(coeff)
