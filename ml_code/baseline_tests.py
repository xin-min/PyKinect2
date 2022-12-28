import gym
from torch.utils.tensorboard import SummaryWriter

from flatland_env import FlatlandEnv

# import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim:int):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "laserscan":
                # n_input_channels = subspace.shape[0]
                self.cnn = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=11, padding='same'),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=5, padding='same'),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=5, stride=5),
                    nn.Conv1d(64,128, kernel_size=5, padding='same'),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=4, stride=4),
                    # nn.ReLU(),
                    nn.Flatten(),
                )

                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = self.cnn(
                        th.as_tensor(observation_space.sample()["laserscan"]).float().reshape((1,1,360))
                    ).shape[1]
                    # print(n_flatten)
                    # if len(n_flatten) < 3:
                    #     n_flatten = n_flatten[0] * n_flatten[1]

                self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
                self.total = nn.Sequential(self.cnn,self.linear)
                # self.total = self.cnn

                extractors[key] = self.total
                total_concat_size += features_dim
            elif key == "desired_goal":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 3),
                    nn.Flatten(start_dim=1)
                )
                total_concat_size += 3

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            input_elem = observations[key]
            if key == "laserscan":
                input_elem = observations[key]
                laser_processed = extractor(input_elem)
                # if len(laser_processed.shape) < 3:
                #     flatten = nn.Flatten(0)
                #     laser_processed = flatten(laser_processed).reshape(1,-1)
                output = laser_processed
            else:
                output = extractor(input_elem)
            encoded_tensor_list.append(output)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        for item in encoded_tensor_list:
            print(item.shape)
        return th.cat(encoded_tensor_list, dim=1)

reward_weights = {
  'heading': 0,
  'dist': 1,
  'time': 0,
}

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=30),
)

env = FlatlandEnv(render_mode='human',sim_factor=5.0, rewards=reward_weights)
# from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO, DDPG, SAC

model = SAC("MultiInputPolicy", env,verbose=1,learning_rate=1e-3,policy_kwargs=policy_kwargs,tensorboard_log='logs')
model = model.learn(total_timesteps=10000,progress_bar=True)

model.save('saved_models')

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    if done:
      obs = env.reset()
