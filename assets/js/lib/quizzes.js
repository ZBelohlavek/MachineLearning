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
          why: 'Without a hidden layer the network is just logistic regression: one weighted sum through one squashing function, and its boundary is always a straight line. A spiral can\'t be cut with a line, so patience, data and tuning are all beside the point. Knowing whether a model is <em>undertrained</em> or simply <em>unable to represent the answer</em> saves you the week you\'d otherwise spend tuning the wrong thing.',
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
          why: 'That divergence is what overfitting looks like. The extra capacity is going into fitting individual training points, noise slider included, and noise doesn\'t generalise by definition. Weight decay, more data, or a smaller network will each pull the two curves back together.',
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
          why: 'The gradient tells you which way is downhill <em>right where you are standing</em>, and nothing more. Multiply it by too big a number and you land somewhere the slope no longer describes, frequently higher than you started. You can watch it on the loss landscape panel: the white path stops descending and starts pinballing.',
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
          why: 'It computes right side minus left side. That\'s zero anywhere the brightness is flat, no matter how bright the region is, and large wherever brightness changes horizontally. The sign carries direction: dark-to-light one way, light-to-dark the other. That\'s why the output is drawn centred on grey instead of starting from black.',
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
          why: 'Each 2×2 block gets replaced by its largest value, so "there\'s a strong vertical edge somewhere around here" survives and "at exactly this pixel" doesn\'t. Discarding that on purpose is what makes the network shrug at a digit drawn two pixels to the left.',
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
          why: 'Two stacked 3×3 layers see a 5×5 neighbourhood using 18 weights per channel pair rather than 25. Three see 7×7. And there\'s a ReLU between each one, so the composition expresses far more than any single large linear filter could. Depth buys receptive field cheaply, which is most of the reason modern vision networks are deep rather than wide.',
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
          why: 'Set augmentation to zero and both sets fill up with the same clean, upright, machine-drawn digits, so the test set certifies nothing whatsoever about handwriting. This is the most common way a real project quietly fails: the benchmark and the deployment differ, and the benchmark has no way of telling you. Turning augmentation up makes training harder, the number lower, and the model better.',
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
          why: 'Nobody told it to build edge detectors. Gradient descent found them because, for a 3×3 window, measuring local change is simply the most useful thing on offer. Humans reached the same conclusion when they designed Sobel kernels by hand decades earlier.',
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
          why: 'Softmax shares certainty out among the ten options it was handed, and "none of these" isn\'t one of them. Draw a spiral and it will still name a digit, sometimes with real confidence. Knowing when a model has left the world it was trained on is an open research problem, not an oversight on this page.',
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
          why: 'It\'s a soft, differentiable lookup. Each token asks a question (its query), every token advertises what it holds (its key), the match scores become softmax weights, and the answer is a weighted blend of everyone\'s values. Nothing in that description mentions images, which is exactly why the same operation runs language models unchanged.',
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
          why: 'A dot product of d-dimensional vectors grows roughly like √d. Leave it unscaled and softmax hardens into a single pick, its gradient goes to almost nothing, and the layer stops learning. One symbol in the formula fixes a real training failure, and it\'s the sort of detail that separates an implementation that trains from one that mysteriously doesn\'t.',
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
          why: 'The CNN starts life already knowing that nearby pixels belong together and that a pattern means the same thing anywhere in the image. The transformer has to learn both from examples, and examples cost data. Move the training-images slider and you can watch that cost directly. Give it hundreds of millions of images and the ranking flips, which is the actual finding of the ViT paper.',
        },
      ],
    },

    language: {
      id: 'language', badge: 'quiz-language',
      title: 'Check yourself · language models',
      questions: [
        {
          q: 'Attention here is <em>causal</em>: position t may look at positions 0 to t and no further. Why does that matter?',
          options: [
            'It makes the model faster',
            'Without it, predicting the next character could simply read the answer from the future',
            'It keeps the attention map symmetric',
            'It reduces the number of parameters',
          ],
          answer: 1,
          why: 'The training signal is "what character comes next", asked at every position simultaneously. Let a position see ahead and it learns to copy the answer instead of predicting it: training loss collapses to nothing and the samples come out as garbage. The triangular attention map on that page is this constraint made visible.',
        },
        {
          q: 'Your model has a 24-character context. What can it never learn?',
          options: [
            'The spelling of common words',
            'That a sentence tends to end in a full stop',
            'That a name mentioned at the start of a paragraph should reappear at the end',
            'Which characters usually follow a space',
          ],
          answer: 2,
          why: 'Anything further back than the window doesn\'t exist as far as the model is concerned. There\'s no memory beyond it. Spelling and local punctuation fit comfortably inside 24 characters; a callback across a paragraph does not. Context length is the most consequential single number in a language model, and it\'s why so much engineering goes into stretching it.',
        },
        {
          q: 'You drop the temperature to 0.1 and the sample becomes one phrase repeated forever. Why?',
          options: [
            'The model has stopped learning',
            'Low temperature sharpens the distribution until it always picks its single most likely character, and that loops',
            'The context window is too short',
            'The vocabulary is too small',
          ],
          answer: 1,
          why: 'Temperature divides the scores before the softmax. Near zero the biggest score wins every single time, so generation turns deterministic, and a deterministic walk through a finite context eventually revisits a state and loops forever. Above 1 the distribution flattens out and the model starts inventing spellings. The readable text lives in between.',
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
          why: 'It\'s a prediction about the entire rest of the episode, not about the next step, which is why a square far from the goal still carries a positive value once some route through it exists. That "and acting sensibly afterwards" clause is doing real work: it\'s the max in the update rule, and it\'s what lets value flow backwards one square per visit.',
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
          why: 'Greedy isn\'t the same as optimal while your estimates are still wrong. Without occasional random moves the agent never collects the evidence that would correct them, so it settles confidently onto whatever it happened to find first. Every RL algorithm has some answer to this, the entropy bonus in the Rocket League lesson included.',
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
          why: 'The agent is being exactly correct about the world it actually lives in. If one move in ten comes out random, standing next to a pit carries a genuine chance of falling in, and the value function is reporting that honestly. It\'s a small clean example of a system optimising for the environment as it is rather than as you pictured it.',
        },
      ],
    },

    connect4: {
      id: 'connect4', badge: 'quiz-connect4',
      title: 'Check yourself \u00b7 search and self-play',
      questions: [
        {
          q: 'You drag thinking time down to 1 simulation. The same trained network now loses to a bot whose whole strategy is "take a win, block a loss". What does that tell you?',
          options: [
            'The network was never trained properly',
            'The network holds strategy, and the search supplies the tactics',
            'One simulation is a bug \u2014 it should still search',
            'The scripted bot is stronger than it looks',
          ],
          answer: 1,
          why: 'At 1 simulation you are playing the raw network, and it plays the opening well while walking into fours it could have seen. Measured on the agent that ships with the page, forty games against that bot each time: 23 wins with no search, 39 with a hundred simulations, identical weights. Asked to find a move that wins immediately, the raw network picks it about a quarter of the time. The network learned what good positions look like across thousands of games; it never learned to check three moves ahead, because it was never asked to. That division of labour is the entire point of the method.',
        },
        {
          q: 'During self-play the agent wins almost exactly half its games, from the first minute to the last. Why is that number useless?',
          options: [
            'It is a bug in how the games are scored',
            'It is both players, so wins and losses always cancel',
            'Connect 4 is a drawn game with perfect play',
            'The agent is not learning',
          ],
          answer: 1,
          why: 'One network plays both colours, so every win it records is also a loss it records. The score is pinned near half regardless of strength, exactly as in the Rocket League lesson where self-play goals for and against track each other. The only measurement that can move is against something that does not improve alongside you \u2014 here, the fixed scripted bot behind the "Test it" button.',
        },
        {
          q: 'In the PUCT formula the search adds c \u00b7 P(a) \u00b7 \u221aN / (1 + N(a)) to each move\u2019s recorded score. What does the (1 + N(a)) on the bottom accomplish?',
          options: [
            'It keeps the numbers small enough to avoid overflow',
            'It makes the search abandon the network\u2019s guess once a move has real results',
            'It stops the same move being played twice in a game',
            'It scales the value head into the right range',
          ],
          answer: 1,
          why: 'A move that has never been tried has N(a) = 0, so the whole term is just the network\u2019s prior and the search goes wherever the network points. Every visit grows the denominator, shrinking the prior\u2019s influence until the recorded results dominate. It is a dial that slides from "trust the model" to "trust the evidence", and it slides on its own as evidence arrives.',
        },
      ],
    },
    wordle: {
      id: 'wordle', badge: 'quiz-wordle',
      title: 'Check yourself \u00b7 information',
      questions: [
        {
          q: 'A guess comes back with the same colouring no matter which of the remaining words turns out to be the answer. How many bits did it buy you?',
          options: [
            'Five, one per letter',
            'Zero',
            'It depends how many letters came back green',
            'One, because you eliminated the guess itself',
          ],
          answer: 1,
          why: 'Information is the removal of uncertainty, so a result you could have predicted in advance carries none of it. Every candidate lands in the same bucket, the list is exactly as long as before, and log\u2082(1) is 0. This is why a word full of repeated letters scores so badly: the second V in VIVID asks a question the first V already answered.',
        },
        {
          q: 'Partway through a game the solver plays a word that the colours have already ruled out \u2014 a word that cannot possibly be the answer. Why would it do that?',
          options: [
            'A bug: it should only ever guess words that can still win',
            'Because that word splits the surviving candidates better than any word that can win',
            'To use up a turn while it thinks',
            'Because unusual letters are always worth trying',
          ],
          answer: 1,
          why: 'Winning this turn and learning the most are two different goals, and they usually point at different words. Giving up a small chance of finishing now can guarantee finishing next turn instead of the turn after. The solver only stops doing this when two or fewer candidates remain, at which point guessing one of them is free.',
        },
        {
          q: 'TRACE buys more information than CRANE and gives a better average, yet CRANE never needs more than four guesses while TRACE sometimes needs five. What does that show?',
          options: [
            'The entropy calculation has an error in it',
            'Optimising the average and optimising the worst case are different problems',
            'CRANE is the better opening word by every measure',
            'Five guesses means the solver failed',
          ],
          answer: 1,
          why: 'Expected information gain is an average, and averages say nothing about tails. A greedy step that is best in expectation can leave a rare, awkward group behind, and a slightly worse step can avoid it. Which one you want depends on whether you mind an occasional bad game \u2014 and deciding that before you start optimising is most of the work.',
        },
      ],
    },
    fighter: {
      id: 'fighter', badge: 'quiz-fighter',
      title: 'Check yourself \u00b7 game theory',
      questions: [
        {
          q: 'The equilibrium agent plays someone who blocks high almost half the time. Over four thousand rounds it earns almost exactly nothing. Why does it not punish the habit?',
          options: [
            'It has not trained for long enough',
            'Punishing means leaving equilibrium, which would make it exploitable in turn',
            'Blocking high is already the strongest move',
            'It cannot see what the opponent played',
          ],
          answer: 1,
          why: 'Unexploitable and winning are different goals. The equilibrium mixture guarantees that nothing you do can take money off it, and the price of that guarantee is that it takes nothing off you either. To punish a habit it would have to lean towards one answer, and the moment it leans, a reply exists that beats it. Poker programs accept this trade on purpose; human professionals usually do not.',
        },
        {
          q: 'Regret matching keeps two things: the mixture implied by current regrets, and the running average of every mixture it has played. Which one is the answer, and why?',
          options: [
            'The current mixture, because it reflects the most recent information',
            'The running average, because that is what converges to unexploitable',
            'Either, since they end up the same',
            'Neither \u2014 it plays the single action with the highest regret',
          ],
          answer: 1,
          why: 'This is the part that catches everyone. The current mixture keeps swinging and is often badly exploitable on its own, which you can watch on the orange line in the self-play chart: it never settles. The guarantee in the theorem is about the average, so the agent samples from the average. Reading the current mixture as the strategy is the most common way to implement this algorithm wrongly.',
        },
        {
          q: 'You beat the reader by establishing a habit for ten rounds and then switching. What does that tell you about best-response agents?',
          options: [
            'That the reader was badly implemented',
            'That any agent committing to a single best answer is itself exploitable',
            'That ten rounds is too short a memory',
            'That exploitative play never works',
          ],
          answer: 1,
          why: 'A best response is a pure strategy, and a pure strategy always has something that beats it \u2014 that is the same fact that makes the game a game in the first place. The reader takes real money off anyone predictable and hands it straight back to anyone who notices what it has settled on. Fighting-game players call the setup and switch "conditioning"; it works on people for exactly the same reason.',
        },
      ],
    },
    kitchen: {
      id: 'kitchen', badge: 'quiz-kitchen',
      title: 'Check yourself \u00b7 cooperation',
      questions: [
        {
          q: 'Two agents are trained by identical self-play, differing only in random seed. Each is good with a copy of itself and much worse with the other. What went wrong?',
          options: [
            'One of them was undertrained',
            'Nothing broke \u2014 each learned a convention that assumed its partner was itself',
            'The random seed was not applied correctly',
            'The reward function was mis-specified',
          ],
          answer: 1,
          why: 'Self-play optimises a pair, and a pair is free to agree on anything: who fetches, who plates, which way round the counter everyone walks. The agreement is worth real points and costs nothing to make, so it gets made \u2014 and it is completely invisible until a partner turns up who was not at the meeting. Both agents are individually fine. What neither has is any reason to be understandable.',
        },
        {
          q: 'You set the entropy bonus to zero and train the kitchen agent for thousands of episodes. It never serves a single soup. Why does this task punish that so severely?',
          options: [
            'Zero entropy makes the gradient undefined',
            'The chain from an empty kitchen to a served soup is six steps, which random play almost never completes',
            'The policy network is too small without it',
            'Entropy is required for the reward shaping to work',
          ],
          answer: 1,
          why: 'Onion, onion, onion, wait, plate, serve. Every link has to be stumbled into before the reward at the end is ever seen, and a policy that stops experimenting stops finding the next link. In a one-step task an exploration bonus is a nicety; in a six-step chain it is the difference between a working agent and a flat line. Measured here: the same settings reach the scripted pair\u2019s score with the bonus on and score zero with it off.',
        },
        {
          q: 'The by-the-book cook is simpler than either trained agent, yet it is the easier partner for a human. Why?',
          options: [
            'It moves faster',
            'It has no private convention to guess, and it routes around whoever is in the way',
            'It was trained on human data',
            'It gets a larger share of the reward',
          ],
          answer: 1,
          why: 'Being predictable is worth more than being strong when someone has to work alongside you. The scripted cook follows the recipe and steps around obstacles, so anything you do is compatible with it. The self-play agents are better in their own company and harder for anyone else to fit around. Anything that has to work with a person \u2014 a car, a negotiator, an assistant \u2014 faces this trade, and the usual fix is to train against a deliberately varied population rather than against yourself.',
        },
      ],
    },
    factory: {
      id: 'factory', badge: 'quiz-factory',
      title: 'Check yourself \u00b7 optimisation without gradients',
      questions: [
        {
          q: 'Twenty thousand random layouts produced not one working factory, while a search that edits one layout repeatedly finds one every time. What is the difference?',
          options: [
            'The search evaluates layouts more accurately',
            'The search keeps partial progress and builds on it; random sampling throws everything away each try',
            'Random sampling did not try enough layouts',
            'The search is allowed a bigger budget',
          ],
          answer: 1,
          why: 'Both get the same simulator and the same budget. Working factories are a vanishingly thin slice of the space, so hitting one by chance is hopeless no matter how many attempts you make \u2014 twenty thousand or twenty million. The annealer never starts from scratch: it holds onto a layout that smelts ore, and looks for edits that add to it. Keeping what works is the entire advantage.',
        },
        {
          q: 'The search reliably reached "makes plates, budget fully spent, no assembler" and stopped improving. What fixed it?',
          options: [
            'A lower temperature',
            'Letting a new machine clear cheap belts elsewhere to pay for itself',
            'More simulation ticks per candidate',
            'A larger grid',
          ],
          answer: 1,
          why: 'It had filled the budget with belts and could no longer afford the nine-cost assembler it needed, and no single edit could change that: removing a belt scored worse, and adding the assembler was refused. A structural change has to be allowed to pay for itself. This is a move-set problem rather than a tuning problem, and no amount of adjusting the annealing schedule would have touched it.',
        },
        {
          q: 'A layout that is one assembler short of working scores exactly what an empty grid scores. What does that tell you about the objective?',
          options: [
            'The simulation is running for too few ticks',
            'It is far too sparse to search directly, so the search needs partial credit',
            'The budget is set too low',
            'Throughput is the wrong thing to want',
          ],
          answer: 1,
          why: 'Throughput is exactly what you want and exactly what you cannot steer by, because until the last piece is in place every partial factory scores zero. The search only works once it is also paid for ore reaching a smelter and plates coming out of one. This is the sparse-reward problem from the Rocket League lesson with no learning anywhere in it \u2014 shaping is not a reinforcement learning trick, it is what you do whenever the thing you want is too rare to aim at.',
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
          why: 'Mutation is blind, and most changes make a car worse. All of the direction comes from which brains are allowed to reproduce. That\'s a far noisier signal than a derivative, which is why it needs a whole population and many generations, and why it falls apart once there are millions of weights to search.',
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
          why: 'Mutation is the only source of new behaviour and also the main source of destruction. Too little and the population improves smoothly, then stalls, stuck within reach of where it started. Too much and it can\'t hold on to anything it finds. It\'s the same exploration-versus-exploitation trade as ε in Q-learning, wearing different clothes.',
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
          why: 'Evolution needs no derivative, so the score can be anything you\'re able to measure: did it finish, did a human prefer it. It also parallelises trivially. What it can\'t do is search a billion-parameter space efficiently, because it collects one number per lifetime instead of one per step.',
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
          why: 'The agent optimises the function you actually wrote, not the one you had in mind. Shaping rewards are how you get it off the ground and also how you accidentally specify a ball-chaser. This is reward hacking in miniature, and the identical failure shows up in far more serious systems dressed in far more complicated language.',
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
          why: 'Winning from a hopeless position ought to teach more than winning from an easy one, and subtracting the critic\'s expectation is what makes that true. It also cuts the variance of the update dramatically, which is the difference between learning in minutes and never converging at all.',
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
          why: 'Every batch is collected by one specific policy and only tells you about behaviour near that policy. Take a huge step on the strength of it and you land somewhere the data says nothing about, usually somewhere worse, with no route back. The clip is a trust region: improve, but not further than this evidence can support.',
        },
      ],
    },
  };

  return { QUIZZES };
});
