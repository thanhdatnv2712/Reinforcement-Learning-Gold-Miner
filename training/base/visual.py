import numpy as np
import matplotlib.pyplot as plt

with open("sta_loss.txt") as f:
    losses = list(map(float, f.read().strip().split('\n')))
print (max(losses))
#print (losses)
plt.plot(losses[70000:], label='l1')
plt.title('Training loss')
plt.xlabel('...')
plt.ylabel('Loss')
#plt.legend()
#plt.show()
plt.savefig('loss.png')
plt.close()

with open("sta_reward.txt") as f:
    reward = list(map(float, f.read().strip().split('\n')))
print (max(reward))
#print (reward)
#print (losses)
plt.plot(reward, label='reward')
plt.title('Reward')
plt.xlabel('...')
plt.ylabel('reward')
#plt.legend()
#plt.show()
plt.savefig('reward.png')
