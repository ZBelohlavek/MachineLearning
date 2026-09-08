# How the writing on this site should sound

The lessons are explanations, not documentation. Someone is reading them at
eleven at night with a canvas full of dots in front of them, and the prose has
to keep them there. Rules, in rough order of how much they matter.

## 1. Sound like a person who ran the experiment

The strongest thing this site has is that the numbers in it are real. The
multi-head attention result, the reward-hacked agent's shadowing behaviour, the
Web Worker measurement — those all came out of actual runs, several of them
surprising. Write them the way you'd tell a colleague, in first person, with the
expectation included:

> I expected four heads to beat one. It lost by twenty points, twice, and I
> still don't have a great story for why.

Not:

> Multi-head attention does not always improve performance at small scale.

First person is reserved for things that actually happened during development.
Don't fake it. If nobody ran an experiment, don't imply someone did.

## 2. Vary the rhythm, and let paragraphs end quietly

The failure mode here isn't bad sentences, it's *identical* sentences: every
paragraph the same length, every one closing on a quotable summary line. Four in
a row and the reader starts hearing the wind-up coming.

Aim for at most one memorable closer per section, not per paragraph. Let the
others just stop. A paragraph is allowed to end on "Then it does it again."

Mix lengths hard. A twelve-word sentence next to a forty-word one reads as a
voice. Two twenty-five-word sentences read as a template.

## 3. Ration the em-dash

Two per page, roughly. They're the right punctuation for a genuine aside, and
the wrong one for a comma, a colon, a full stop, or a parenthesis, which is what
they usually get used for. When you catch one, ask which of those four it
actually wanted to be.

## 4. Avoid the antithesis reflex

"Not X, but Y." "It isn't A; it's B." Once a lesson is fine. It's a real
rhetorical figure and it does real work when the contrast is the point. As a
default sentence shape it turns every idea into a correction of an idea nobody
had.

## 5. Contractions, second person, plain verbs

"You'll get", "it doesn't", "here's". Address the reader directly and give them
things to do. Prefer the concrete verb to the abstract noun: "the agent stops
exploring", not "a reduction in exploratory behaviour occurs".

## 6. Specifics beat adjectives

"Slow" is nothing. "About four minutes on a 2021 laptop" is something. Any time
a sentence reaches for an intensifier, see whether a measurement fits instead.

## 7. Don't reuse the same sentence on eight pages

Section furniture (quiz intros, experiment headers) should be written fresh per
lesson. Identical boilerplate repeated across the site is the single most
obvious sign that nobody was really writing.

## 8. Humour is allowed, jokes are not

A dry aside in passing is good. A setup-and-punchline stops the lesson dead and
ages badly on the second read. "The car spends its first thirty seconds doing
donuts" is funny enough and is also just true.
