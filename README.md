# session10_endgame_final
Final Project of EVA2 - self driving car using TD3

Following are the main design points:

- Actor / Actor target:

  - Simple conv network that can do ~99 % on MNIST with Adam optimizer.
  - Input is 80x80 image around the car
  - Output is one action that is used to calculate the angle car should move
  
  - Critic / Critic target
  
    - Combination of conv and fully connected laters. First perform conv until output of 7x7 size is obtained.
      Then cat the action to it and run a 2 layer FC to get 1 (Q) output.
      
- The input image used as input data is obtained by taking out 80 pixels around the car's coordinate on sand. The cut out is always horizontal to the x-axis. The image is all white except the sand portion which is black. It is always 80x80 even if the car is in the corner and sufficient space is not available. All the outside area appear white in the image.
- A line was also added on the image to show where the destination is. Thought it will act as "orientation". But, this did not turn out to be very useful. Could not test extensively as i was trying to test it on my Windows laptop without a GPU card.

Parameters:

- Do not take action from Actor network before 10,000 steps.
- Every 1000 steps, a done = True was forced so that training can be performed
- Randomly selected bacth of 100 is chosen to train the network
- The game runs for ever from scheduler context that is scheduled 60 times in a second
- Two destination. Car should swap between them. done is true when it reaches its destination. But, env is not reset when done is true.
- Reward discount factor is 0.99
- target n/w udate rate (tau) is 0.5
- max action is 1
- Policy noise is 0.2 and noise clip 0.5
- Rewards policy is: -1 if not on road. If on road, live in penalty is -0.2 but if less than 25 units from destination then the reward is 0.1

Approach / Results:

- First tried TD3 with sensor data input. That works fine.
- With CNN added, the testing was limited. So, tested if CNN works without TD3. It did not work very well. But, I guess without TD3, it may not achieve very good results anyway.
- Final code was taking hours without yielding any good results. Ran the final code on friend's computer to get the video.

