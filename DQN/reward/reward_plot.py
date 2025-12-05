import numpy as np
import matplotlib.pyplot as plt


rewards = np.load('rewards.npy') 

print(f"Rewards shape: {rewards.shape}")
print(f"Rewards data type: {rewards.dtype}")


plt.plot(rewards)
plt.title('Feature Rewards')
plt.xlabel('Feature Index')
plt.ylabel('Reward Value')
plt.grid(True)

plt.savefig('feature_rewards_plot.png')  
plt.close()