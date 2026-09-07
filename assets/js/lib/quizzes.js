/* ==========================================================================
   quizzes.js — the questions for each lesson's checkpoint.

   Every question is about something the reader has just made happen on the
   page, and every explanation says why the wrong answer is tempting. They are
   here in one file so they can be read, argued with and improved together.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const QUIZZES = {
    foundations: {
      id: 'foundations', badge: 'quiz-foundations',
      title: 'Check yourself · foundations',
      questions: [
        {
          q: 'You set hidden layers to zero and train on the spiral for as long as you like. Accuracy sticks near 50%. What is actually wrong?',
          options: [
            'The learning rate is too low',
            'With no hidden layer the model can only draw a straight boundary',
            'There is not enough training data',
            'The batch size is too large',
          ],
          answer: 1,
          why: 'Without a hidden layer the network is logistic regression: a weighted sum through one squashing function, whose boundary is always a straight line. A spiral cannot be split by a line, so no amount of patience, data or tuning helps. This is the difference between a model that is <em>undertrained</em> and one that <em>cannot represent the answer</em> — and it is worth being able to tell them apart before you spend a week tuning.',
        },
        {
          q: 'Training loss keeps falling while test loss turns and climbs. What is the network doing?',
          options: [
            'Learning more slowly than before',
            'Hitting a bug in the optimiser',
            'Memorising the noise in the training points rather than the pattern',
            'Running out of capacity',
          ],
          answer: 2,
          why: 'That divergence is the definition of overfitting. The extra capacity is being spent fitting individual training points — including the noise you added with the slider — which by definition does not transfer to points it has never seen. Weight decay, more data or a smaller network all pull the curves back together.',
        },
        {
          q: 'You raise the learning rate from 0.03 to 0.3 and the loss curve turns into a saw. Why?',
          options: [
            'Each step is so large it jumps past the bottom of the valley and back up the other side',
            'The gradients are being computed incorrectly at high learning rates',
            'The network is exploring, which is good',
            'The data is being reshuffled too often',
          ],
          answer: 0,
          why: 'The gradient only tells you which way is downhill <em>right here</em>. Multiply it by too big a number and you land somewhere the slope no longer applies, often further up than you started. Watch it on the loss landscape panel: the white path stops descending and starts pinballing.',
        },
      ],
    },

    convolutions: {
      id: 'convolutions', badge: 'quiz-convolutions',
      title: 'Check yourself · convolutions',
      questions: [
        {
          q: 'A kernel has −1 down its left column and +1 down its right. Where does it respond most strongly?',
          options: [
            'Anywhere the image is bright',
            'On a horizontal edge',
            'On a vertical edge, and its sign says which side is brighter',
            'On the smoothest part of the image',
          ],
          answer: 2,
          why: 'It computes "right side minus left side", which is zero anywhere the brightness is flat — however bright that region is — and large wherever brightness changes horizontally. The sign carries the direction: dark-to-light gives one sign, light-to-dark the other. That is why the output is drawn centred on grey rather than starting at black.',
        },
        {
          q: 'What does max pooling deliberately destroy?',
          options: [
            'The strength of the response',
            'Exactly where in each 2×2 block the response happened',
            'The number of channels',
            'The sign of the response',
          ],
          answer: 1,
          why: 'Each 2×2 block is replaced by its largest value, so "there is a strong vertical edge in this neighbourhood" survives while "at precisely this pixel" does not. Throwing that away on purpose is what makes the network tolerant of a digit drawn a couple of pixels to the left.',
        },
        {
          q: 'Why do real networks stack many 3×3 convolutions instead of using one large kernel?',
          options: [
            'Small kernels are more accurate at every scale',
            'Large kernels cannot be trained by backpropagation',
            'Stacking small kernels sees a wide area for far fewer weights and adds a non-linearity at each step',
            'It is a historical convention with no real benefit',
          ],
          answer: 2,
          why: 'Two stacked 3×3 layers see a 5×5 neighbourhood using 18 weights per channel pair instead of 25, three see 7×7, and so on — with a ReLU in between each, so the composition can express far more than one big linear filter. Depth buys receptive field cheaply, and that is most of why modern vision networks are deep rather than wide.',
        },
      ],
    },

    cnn: {
      id: 'cnn', badge: 'quiz-cnn',
      title: 'Check yourself · convolutional networks',
      questions: [
        {
          q: 'Your model reports 99% test accuracy, then fails badly on the digits you draw. What is the most likely explanation?',
          options: [
            'The test set was drawn from the same narrow distribution as the training set, so it never tested this',
            'The model is broken and the accuracy number is wrong',
            'Your drawing is genuinely not a digit',
            'The softmax is miscalibrated',
          ],
          answer: 0,
          why: 'Set augmentation to zero and both sets contain the same clean, upright, machine-drawn digits — so the test set certifies nothing about handwriting. This is the most common way a real project goes wrong: the benchmark and the deployment differ, and the benchmark has no way to tell you. Turning augmentation up makes training harder and the number lower, and the model genuinely better.',
        },
        {
          q: 'After training, the first layer\'s learned 3×3 kernels mostly resemble…',
          options: [
            'Random noise, since they are initialised randomly',
            'Small pictures of digits',
            'Edge and stroke detectors, much like the Sobel kernels you wrote by hand',
            'Blur filters',
          ],
          answer: 2,
          why: 'Nobody told it to build edge detectors. Gradient descent found them because, for a 3×3 window, measuring local change is simply the most useful thing available — the same conclusion humans reached when they designed Sobel kernels by hand decades earlier.',
        },
        {
          q: 'The ten confidence bars are a softmax, so they always add to 100%. What can this model therefore never tell you?',
          options: [
            'Which digit is most likely',
            'That the thing you drew is not a digit at all',
            'How confident it is between two digits',
            'Which pixels mattered',
          ],
          answer: 1,
          why: 'Softmax distributes certainty among the ten options it was given; "none of these" is not one of them. Draw a spiral and it will still name a digit, sometimes confidently. Knowing when a model is outside the world it was trained on is a genuine open problem, not an oversight in this page.',
        },
      ],
    },

    attention: {
      id: 'attention', badge: 'quiz-attention',
      title: 'Check yourself · attention',
      questions: [
        {
          q: 'In one sentence, what does an attention layer compute?',
          options: [
            'A convolution with learned kernel sizes',
            'For each token, a weighted average of every token\'s value, where the weights come from how well queries match keys',
            'The most similar pair of patches in the image',
            'A ranking of patches by brightness',
          ],
          answer: 1,
          why: 'It is a soft, differentiable lookup: each token asks a question (its query), every token advertises what it holds (its key), the match scores become softmax weights, and the answer is the weighted mix of everyone\'s values. Nothing in that description mentions images, which is exactly why the same operation powers language models.',
        },
        {
          q: 'Why are the attention scores divided by √d before the softmax?',
          options: [
            'To make the maths match the convolution formula',
            'To normalise for different image sizes',
            'Because dot products grow with dimension, and large scores make softmax saturate and its gradient vanish',
            'It is an arbitrary constant that happens to work',
          ],
          answer: 2,
          why: 'A dot product of d-dimensional vectors grows roughly like √d. Left unscaled, the softmax turns into a hard pick, its gradient goes to nearly zero and the layer stops learning. One symbol in the formula fixes a real training failure — and it is the kind of detail that separates an implementation that trains from one that mysteriously does not.',
        },
        {
          q: 'On this page the CNN reaches 99% while the transformer struggles past 85% on the same digits. Why?',
          options: [
            'The transformer is a worse architecture',
            'Attention cannot represent what convolution can',
            'The CNN has more parameters',
            'Convolution comes with locality and weight sharing built in, and those assumptions are worth a great deal when data is scarce',
          ],
          answer: 3,
          why: 'The CNN starts life knowing that nearby pixels belong together and that a pattern means the same thing anywhere in the image. The transformer must learn both from examples, which costs data — you can watch that cost directly by moving the training-images slider. Give it hundreds of millions of images and the ranking reverses, which is the actual finding of the ViT paper.',
        },
      ],
    },

    gridworld: {
      id: 'gridworld', badge: 'quiz-gridworld',
      title: 'Check yourself · Q-learning',
      questions: [
        {
          q: 'What does the number Q(s, a) actually mean?',
          options: [
            'The reward you get immediately for taking action a in state s',
            'The probability of taking action a in state s',
            'The total future reward expected from taking a in s and acting sensibly afterwards',
            'The distance from s to the goal',
          ],
          answer: 2,
          why: 'It is a prediction about the whole rest of the episode, not about the next step — which is why a square far from the goal still gets a positive value once a route through it exists. That "and acting sensibly afterwards" is doing real work: it is the max in the update rule, and it is what lets value flow backwards one square per visit.',
        },
        {
          q: 'You set ε to 0 before the agent has explored. What goes wrong?',
          options: [
            'It always picks the move it currently thinks is best, so the first mediocre route it stumbles into becomes the only one it ever considers',
            'It stops learning entirely',
            'It moves randomly forever',
            'The Q-values overflow',
          ],
          answer: 0,
          why: 'Greedy is not the same as optimal when your estimates are still wrong. Without occasional random moves the agent never gathers the evidence that would correct them, so it converges confidently onto whatever it happened to find first. Every RL algorithm has some answer to this, including the entropy bonus in the Rocket League lesson.',
        },
        {
          q: 'On the cliff track, raising ε makes the agent walk the long way round instead of hugging the edge. Is it being timid?',
          options: [
            'Yes — exploration makes it behave irrationally',
            'No — with random moves in the mix, a square next to a pit genuinely has a lower expected value',
            'No — it simply has not learned the short route yet',
            'Yes — the discount factor is too low',
          ],
          answer: 1,
          why: 'The agent is being exactly correct about the world it actually lives in. If one move in ten is random, standing next to a pit carries a real chance of falling in, and the value function reflects that. It is a small, clean example of a system optimising for the environment as it is rather than as you imagined it.',
        },
      ],
    },

    racer: {
      id: 'racer', badge: 'quiz-racer',
      title: 'Check yourself · evolution',
      questions: [
        {
          q: 'There is no gradient anywhere in this lesson. What plays the role that the gradient plays in the other lessons?',
          options: [
            'The sensors, which point the car in the right direction',
            'Selection: only the brains that got furthest are copied into the next generation',
            'The random mutations, which are always improvements',
            'The fitness function, which is differentiated numerically',
          ],
          answer: 1,
          why: 'Mutation is blind — most changes make a car worse. The direction comes entirely from which brains get to reproduce. It is a much noisier signal than a derivative, which is why it needs a whole population and many generations, and why it stops scaling once there are millions of weights to search.',
        },
        {
          q: 'You set the mutation strength to 0.6 and the best-car line starts sawing up and down. Why?',
          options: [
            'The population is too small to average out noise',
            'The track is too hard at that setting',
            'Children land so far from their parents that a good driver is destroyed as fast as one appears',
            'Elitism is switched off at high mutation',
          ],
          answer: 2,
          why: 'Mutation is the only source of new behaviour and the main source of destruction. Too little and the population improves smoothly then stalls, unable to reach anything far from where it started; too much and it cannot hold on to what it found. It is the same exploration-versus-exploitation trade as ε in Q-learning, wearing different clothes.',
        },
        {
          q: 'When would you reach for evolution over policy gradients on a real problem?',
          options: [
            'When the model has few parameters and the objective is messy or non-differentiable',
            'When you have a very large network and lots of data',
            'When you need the best possible sample efficiency',
            'When the environment is fully observable',
          ],
          answer: 0,
          why: 'Evolution needs no derivative, so the score can be anything you can measure — "did it finish", "did a human prefer it" — and it parallelises trivially. What it cannot do is search a billion-parameter space efficiently, because it only learns one number per lifetime instead of one per step.',
        },
      ],
    },

    rocket: {
      id: 'rocket', badge: 'quiz-rocket',
      title: 'Check yourself · policy gradients',
      questions: [
        {
          q: 'You set the touch reward high and the goal reward near zero, then train. What do you get?',
          options: [
            'A faster learner, since touches happen more often',
            'An agent that follows the ball around and rarely scores',
            'The same agent, since goals follow from touches anyway',
            'An agent that refuses to move',
          ],
          answer: 1,
          why: 'The agent optimises the function you actually wrote, not the one you meant. Shaping rewards are how you get it off the ground, and they are also how you accidentally specify a ball-chaser. This is reward hacking in miniature, and the same failure appears in far more serious systems dressed in far more complicated language.',
        },
        {
          q: 'The critic predicts V(s). What is the advantage, and why use it instead of the raw return?',
          options: [
            'The total reward collected — it is simpler to compute',
            'The reward for the current step only — it reduces memory use',
            'How much better things turned out than the critic expected — so credit goes to decisions that beat expectations, not to lucky starting positions',
            'The gradient of the reward with respect to the action',
          ],
          answer: 2,
          why: 'Winning from a hopeless position should teach far more than winning from an easy one, and subtracting the critic\'s expectation is what makes that true. It also dramatically reduces the variance of the update, which is the difference between learning in minutes and never converging.',
        },
        {
          q: 'Why does PPO clip how far the policy may move in one update?',
          options: [
            'To keep the weights small, like weight decay',
            'To stop one noisy batch from moving the policy somewhere it cannot recover from, since the data was collected by the old policy',
            'To make the algorithm run faster',
            'To keep the entropy high',
          ],
          answer: 1,
          why: 'Every batch is collected by one specific policy, and it only tells you about behaviour near that policy. Take a huge step on the strength of it and you land somewhere the data says nothing about — often somewhere worse, with no way back. The clip is a trust region: improve, but not further than this evidence supports.',
        },
      ],
    },
  };

  return { QUIZZES };
});
