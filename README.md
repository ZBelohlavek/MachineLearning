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
| **[Q-Learning](lessons/gridworld.html)** | Build a maze, watch a value table fill in square by square, then race the agent through it | States, actions, rewards, the Bellman update, exploration vs exploitation |
| **[Evolve a driver](lessons/evolve-a-driver.html)** | Watch sixty cars learn to drive a track you drew, then race the champion | Neuroevolution, fitness functions, selection and mutation, gradient-free learning |
| **[Rocket League Bot](lessons/rocket-league.html)** | Design the reward function and train a self-play agent to score | Policy gradients, PPO, advantage estimation, reward shaping, self-play |
| **[Final challenge](lessons/final-challenge.html)** | Pick the right approach for nine situations, then diagnose four training runs from their curves | Judgement: matching method to problem, and reading failure from a chart |

The flagship is the last one: a top-down arena where a policy-gradient agent learns to drive, hit the
ball and score by playing against itself. You choose the reward weights, the opponent and the
hyperparameters, watch the training live, and then play against the result with the arrow keys.

## Nothing to wait for

Training from scratch is the honest experience, but it asks for patience before anything interesting
happens, so the site ships pre-trained models and lets you skip ahead:

- **A training timelapse.** Eleven checkpoints from a full 12,000-episode Rocket League run. Drag
  the slider and each one plays live in the arena — random twitching, first accidental touches,
  driving at the ball, aiming at the goal — with the recorded curve marking where you are. One
  click loads any of them into the trainer to continue from.
- **A pre-trained CNN** (99.3% test accuracy), so the drawing pad recognises your handwriting the
  moment the page opens.
- **A pre-trained Vision Transformer** (88.0%), so its attention maps show real structure rather
  than noise before you have trained anything.

Every one of them can be thrown away with a single button to get the full from-scratch experience,
and badges only unlock for models you trained yourself.

## Goals, challenges and quizzes

Twenty-six badges, tracked in `localStorage`, listed inside each lesson as you work and summarised
on the home page. None are for clicking around — each falls into one of three kinds:

- **Things your model does.** Evolve a car that finishes the course, train a bot that beats the
  scripted opponent, catch a network overfitting, score 8 in the 60-second digit duel. Loading a
  pre-trained model does not count; badges are only for runs you trained.
- **Auto-graded challenges.** "Type a kernel that finds horizontal edges" is checked by correlating
  your kernel's output against a Sobel-Y reference, so any scaling or sign flip of a real edge
  detector passes and clicking the preset does not. "Solve the spiral with 8 hidden neurons or
  fewer" is checked the same way, against held-out accuracy.
- **A capstone.** Nine situations where one method fits and three are things people genuinely reach
  for and regret, then four training runs shown only as curves — each the signature of a failure you
  produced yourself earlier in the course. It is the only part of the site that tests whether the
  ideas transfer.
- **Quiz checkpoints.** Three questions at the end of every lesson, about what you just made happen
  rather than about terminology. The explanation appears whichever answer you pick, and says why the
  wrong answer was tempting — the questions live in `assets/js/lib/quizzes.js` so they can be read
  and argued with as a set.

## What's actually in here

All of the machine learning is written from scratch in plain JavaScript, and it's meant to be read:

```
assets/models/
  rocket-stages.js  11 policy checkpoints from a full self-play training run
  cnn-digits.js     a CNN trained to 99.3% on generated digits
  vit-digits.js     a small ViT trained to 88.0% on the same data

tools/
  train-rocket.cjs  trains the agent offline and writes the checkpoints
  train-vision.cjs  trains the CNN and ViT in headless Chromium and exports them

assets/js/lib/
  nn.js       Dense layers, backpropagation, Adam, softmax, seedable RNG
  cnn.js      Conv2D, ReLU, MaxPool2, Flatten, FC — with hand-written backward passes
  vit.js      Token linear layers, single-head self-attention, LayerNorm, a small ViT
  digits.js   A digit dataset generated with the canvas text API
  plot.js     Canvas line charts and colour maps
  ui.js       Shared page chrome, sliders, pills, stat tiles, badges, quizzes
  quizzes.js  The three questions and explanations for each lesson

assets/js/lessons/
  rocket-env.js     Arena physics, egocentric observations, configurable rewards
  rocket-train.js   PPO with GAE, minibatching, curriculum learning, self-play
  rocket.js         Rendering, training UI and the policy inspector
  gridworld.js      Tabular Q-learning on an editable grid
  foundations.js    2D classification playground
  convolutions.js   Interactive kernel explorer
  cnn-digits.js     CNN training, feature maps, the drawing pad and the digit duel
  vit-lesson.js     Patch embeddings, attention maps, position embeddings
  racer.js          Track mask, BFS distance field, sensors and neuroevolution
  capstone.js       The final challenge: scenario picking and curve diagnosis
```

Regenerating the shipped models:

```bash
node tools/train-rocket.cjs 12000          # ~11 minutes, no dependencies
node tools/train-vision.cjs 100            # needs Playwright
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
  League agent starts hitting the ball within a minute or two and scoring shortly after. The
  pre-trained models are a shortcut, never a substitute — each lesson still trains its own model in
  front of you.
- **Models ship as `.js`, not `.json`.** Browsers block `fetch()` on `file://` URLs, and this site is
  meant to work when you double-click `index.html`, so each model assigns a global from a plain
  `<script>` tag.
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

On touch devices the three driving games (Rocket League, the racer and the Q-learning maze) show an
on-screen direction pad, since they would otherwise be unplayable without a keyboard, and the
navigation collapses into a single swipeable row. Canvases you draw on capture touch gestures;
everything else lets a finger scroll past.
