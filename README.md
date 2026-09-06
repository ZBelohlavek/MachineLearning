# Learn ML by Building

An interactive machine learning course that runs entirely in the browser. Six lessons, each one a
working model you train yourself — no servers, no dependencies, no build step, and nothing
pre-trained. Every model on the site starts from random weights while you watch.

**Open `index.html` in a browser.** That's the whole setup. It works straight from disk (`file://`)
or from any static server.

```bash
git clone https://github.com/ZBelohlavek/MachineLearning.git
cd MachineLearning
python3 -m http.server 8000     # then open http://localhost:8000
```

## The lessons

| Lesson | What you do | What it teaches |
| --- | --- | --- |
| **[Foundations](lessons/foundations.html)** | Draw a dataset, configure a network, watch the decision boundary bend | Backpropagation, gradient descent, activations, capacity, overfitting |
| **[Convolutions](lessons/convolutions.html)** | Slide a 3×3 kernel across an image and see every multiplication | Kernels, feature maps, padding, stride, pooling |
| **[Train a CNN](lessons/cnn-digits.html)** | Train a convolutional net on generated digits, then draw one yourself | Conv layers, augmentation, confusion matrices, learned filters |
| **[Vision Transformers](lessons/vision-transformer.html)** | Cut an image into patches and train a real attention model | Patch embeddings, queries/keys/values, attention maps, position embeddings |
| **[Q-Learning](lessons/gridworld.html)** | Build a maze and watch a value table fill in square by square | States, actions, rewards, the Bellman update, exploration vs exploitation |
| **[Rocket League Bot](lessons/rocket-league.html)** | Design the reward function and train a self-play agent to score | Policy gradients, PPO, advantage estimation, reward shaping, self-play |

The flagship is the last one: a top-down arena where a policy-gradient agent learns to drive, hit the
ball and score by playing against itself. You choose the reward weights, the opponent and the
hyperparameters, watch the training live, and then play against the result with the arrow keys.

## What's actually in here

All of the machine learning is written from scratch in plain JavaScript, and it's meant to be read:

```
assets/js/lib/
  nn.js       Dense layers, backpropagation, Adam, softmax, seedable RNG
  cnn.js      Conv2D, ReLU, MaxPool2, Flatten, FC — with hand-written backward passes
  vit.js      Token linear layers, single-head self-attention, LayerNorm, a small ViT
  digits.js   A digit dataset generated with the canvas text API
  plot.js     Canvas line charts and colour maps
  ui.js       Shared page chrome, sliders, pills, stat tiles

assets/js/lessons/
  rocket-env.js     Arena physics, egocentric observations, configurable rewards
  rocket-train.js   PPO with GAE, minibatching, curriculum learning, self-play
  rocket.js         Rendering, training UI and the policy inspector
  gridworld.js      Tabular Q-learning on an editable grid
  foundations.js    2D classification playground
  convolutions.js   Interactive kernel explorer
  cnn-digits.js     CNN training, feature maps and the drawing pad
  vit-lesson.js     Patch embeddings, attention maps, position embeddings
```

Every backward pass — dense, convolution, max pooling, softmax attention and layer norm — was checked
against numerical gradients before being used. If you have only ever called `loss.backward()`, these
files are a short read that shows what it does.

## Tests

```bash
node tests/gradcheck.cjs     # no dependencies
```

Checks every hand-written backward pass — dense layers, convolution, max pooling, softmax attention
and layer norm — against numerical gradients, then sanity-checks the arena physics and confirms that
PPO measurably improves play in twenty seconds. Two details are worth knowing if you read it: the
pass criterion is a directional derivative over all parameters at once (thousands of parameters give
a much better signal-to-noise ratio than perturbing weights one at a time), and per-weight
comparisons skip gradients sitting at the float32 noise floor as well as ReLU kinks, where a finite
difference measures a corner rather than a slope.

```bash
npm i -D playwright && npx playwright install chromium
node tests/browser.cjs       # needs Playwright; the site itself does not
```

Loads every lesson, trains each model, drives the controls (keyboard driving in the arena, painting
the maze, editing kernels, switching architectures) and fails on any console error.

## Design notes

- **No frameworks and no build step.** Each library file works either as a plain `<script>` (exports
  land on `window.ML`) or through `require()` in Node, which is how the gradient checks run.
- **Everything trains live.** The CNN reaches ~99% test accuracy in about twenty seconds; the Rocket
  League agent starts hitting the ball within a minute or two and scoring shortly after.
- **Deliberate failure modes.** Every lesson ends with experiments designed to break the model in an
  instructive way: reward hacking, overfitting, collapsed exploration, learning rates that explode.

## Publishing it

The site is plain static files, so GitHub Pages serves it as-is: in the repository's
**Settings → Pages**, set the source to the branch and the `/ (root)` folder. No build step, no
workflow, no configuration file.

## Browser support

Any modern browser. The heavy lessons are CPU-bound single-threaded JavaScript, so a desktop machine
gives noticeably faster training than a phone. Everything is drawn with the 2D canvas API — no WebGL
and no WebAssembly.
