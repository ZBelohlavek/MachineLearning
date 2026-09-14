/* ==========================================================================
   wordle-words.js — a list of common five-letter English words.

   The real game uses about 2,300 answers and 13,000 allowed guesses. This is a
   smaller, hand-checked list, which keeps the file readable and every entropy
   calculation on the page fast enough to run while you watch. The lesson is
   about how to choose a guess, and that reasoning does not change with the size
   of the list.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const WORDS = [
    'about',  'above',  'abuse',  'actor',  'acute',  'admit',  'adopt',  'adult',  'after',  'again',  'agent',  'agree',
    'ahead',  'alarm',  'album',  'alert',  'alike',  'alive',  'allow',  'alone',  'along',  'alter',  'among',  'anger',
    'angle',  'angry',  'apart',  'apple',  'apply',  'arena',  'argue',  'arise',  'armed',  'array',  'arrow',  'aside',
    'asset',  'audio',  'audit',  'avoid',  'awake',  'award',  'aware',  'badly',  'baker',  'bases',  'basic',  'basis',
    'beach',  'began',  'begin',  'begun',  'being',  'below',  'bench',  'birth',  'black',  'blame',  'blank',  'blast',
    'blind',  'block',  'blood',  'board',  'boost',  'booth',  'bound',  'brain',  'brand',  'brass',  'brave',  'bread',
    'break',  'breed',  'brief',  'bring',  'broad',  'broke',  'brown',  'brush',  'build',  'built',  'buyer',  'cable',
    'carry',  'catch',  'cause',  'chain',  'chair',  'chalk',  'chaos',  'charm',  'chart',  'chase',  'cheap',  'check',
    'chest',  'chief',  'child',  'china',  'chose',  'civil',  'claim',  'class',  'clean',  'clear',  'clerk',  'click',
    'cliff',  'climb',  'clock',  'close',  'cloth',  'cloud',  'coach',  'coast',  'could',  'count',  'court',  'cover',
    'crack',  'craft',  'crane',  'crash',  'crazy',  'cream',  'crime',  'cross',  'crowd',  'crown',  'crude',  'curve',
    'cycle',  'daily',  'dance',  'dated',  'dealt',  'death',  'debut',  'decay',  'delay',  'dense',  'depth',  'doing',
    'doubt',  'dozen',  'draft',  'drain',  'drama',  'drank',  'drawn',  'dream',  'dress',  'dried',  'drill',  'drink',
    'drive',  'drove',  'dying',  'eager',  'early',  'earth',  'eight',  'elect',  'elite',  'empty',  'enemy',  'enjoy',
    'enter',  'entry',  'equal',  'error',  'event',  'every',  'exact',  'exist',  'extra',  'faith',  'false',  'fault',
    'favor',  'feast',  'fence',  'fever',  'field',  'fifth',  'fifty',  'fight',  'final',  'first',  'flame',  'flash',
    'fleet',  'flesh',  'float',  'flood',  'floor',  'flour',  'fluid',  'focus',  'force',  'forth',  'forty',  'forum',
    'found',  'frame',  'fraud',  'fresh',  'front',  'frost',  'fruit',  'fully',  'funny',  'giant',  'given',  'glass',
    'globe',  'glory',  'going',  'grace',  'grade',  'grain',  'grand',  'grant',  'grape',  'grasp',  'grass',  'grave',
    'great',  'green',  'greet',  'grief',  'gross',  'group',  'grown',  'guard',  'guess',  'guest',  'guide',  'happy',
    'harsh',  'haste',  'heart',  'heavy',  'hedge',  'hello',  'hence',  'hobby',  'holds',  'honey',  'honor',  'horse',
    'hotel',  'house',  'human',  'humor',  'hurry',  'ideal',  'image',  'imply',  'index',  'inner',  'input',  'irony',
    'issue',  'joint',  'judge',  'juice',  'known',  'label',  'labor',  'large',  'laser',  'later',  'laugh',  'layer',
    'learn',  'lease',  'least',  'leave',  'legal',  'lemon',  'level',  'light',  'limit',  'lined',  'links',  'liver',
    'lives',  'local',  'lodge',  'logic',  'loose',  'lower',  'loyal',  'lucky',  'lunch',  'lying',  'magic',  'major',
    'maker',  'march',  'match',  'maybe',  'mayor',  'meant',  'medal',  'media',  'mercy',  'merge',  'merit',  'metal',
    'meter',  'micro',  'might',  'minor',  'minus',  'mixed',  'model',  'money',  'month',  'moral',  'motor',  'mount',
    'mouse',  'mouth',  'movie',  'music',  'naked',  'nasty',  'naval',  'needs',  'nerve',  'never',  'newly',  'night',
    'noise',  'noted',  'novel',  'nurse',  'ocean',  'offer',  'often',  'onion',  'order',  'other',  'ought',  'ounce',
    'outer',  'owned',  'owner',  'paint',  'panel',  'panic',  'paper',  'party',  'pasta',  'patch',  'pause',  'peace',
    'peach',  'pearl',  'pedal',  'penny',  'phase',  'phone',  'photo',  'piano',  'piece',  'pilot',  'pitch',  'pivot',
    'place',  'plain',  'plane',  'plant',  'plate',  'plaza',  'point',  'polar',  'porch',  'pound',  'power',  'press',
    'price',  'pride',  'prime',  'print',  'prior',  'prize',  'probe',  'promo',  'proof',  'proud',  'prove',  'pulse',
    'punch',  'pupil',  'purse',  'queen',  'query',  'quest',  'queue',  'quick',  'quiet',  'quilt',  'quite',  'quota',
    'radar',  'radio',  'raise',  'ralph',  'range',  'rapid',  'ratio',  'reach',  'react',  'ready',  'realm',  'rebel',
    'refer',  'reign',  'relax',  'relay',  'remix',  'renew',  'reply',  'rider',  'ridge',  'rifle',  'right',  'rigid',
    'rival',  'river',  'roast',  'robot',  'rocky',  'roman',  'rough',  'round',  'route',  'royal',  'rugby',  'rural',
    'saint',  'salad',  'sales',  'salon',  'sauce',  'scale',  'scene',  'scope',  'score',  'scout',  'scrap',  'screw',
    'sense',  'serve',  'setup',  'seven',  'shade',  'shaft',  'shake',  'shall',  'shame',  'shape',  'share',  'sharp',
    'sheep',  'sheet',  'shelf',  'shell',  'shift',  'shine',  'shirt',  'shock',  'shoot',  'shore',  'short',  'shown',
    'sides',  'sight',  'silly',  'since',  'siren',  'sixth',  'skill',  'skirt',  'slate',  'sleep',  'slice',  'slide',
    'slope',  'small',  'smart',  'smell',  'smile',  'smoke',  'snack',  'snake',  'sneak',  'solar',  'solid',  'solve',
    'sorry',  'sound',  'south',  'space',  'spare',  'spark',  'speak',  'speed',  'spell',  'spend',  'spent',  'spice',
    'spike',  'spine',  'spite',  'split',  'spoke',  'spoon',  'sport',  'spray',  'squad',  'stack',  'staff',  'stage',
    'stain',  'stair',  'stake',  'stand',  'stare',  'start',  'state',  'steam',  'steel',  'steep',  'steer',  'stick',
    'still',  'stock',  'stole',  'stone',  'stood',  'stool',  'store',  'storm',  'story',  'stove',  'strap',  'straw',
    'strip',  'stuck',  'study',  'stuff',  'style',  'sugar',  'suite',  'super',  'surge',  'sweat',  'sweep',  'sweet',
    'swift',  'swing',  'sword',  'table',  'taken',  'taste',  'teach',  'teams',  'tempo',  'tenth',  'thank',  'theft',
    'their',  'theme',  'there',  'these',  'thick',  'thief',  'thing',  'think',  'third',  'those',  'three',  'threw',
    'throw',  'thumb',  'tiger',  'tight',  'timer',  'tired',  'title',  'toast',  'today',  'token',  'tooth',  'topic',
    'torch',  'total',  'touch',  'tough',  'tower',  'toxic',  'trace',  'track',  'trade',  'trail',  'train',  'trait',
    'trash',  'treat',  'trend',  'trial',  'tribe',  'trick',  'tried',  'tries',  'truck',  'truly',  'trunk',  'trust',
    'truth',  'tulip',  'tumor',  'tutor',  'twice',  'twist',  'ultra',  'uncle',  'under',  'union',  'unite',  'unity',
    'until',  'upper',  'upset',  'urban',  'usage',  'usual',  'vague',  'valid',  'value',  'valve',  'vapor',  'vault',
    'venue',  'verse',  'video',  'villa',  'vinyl',  'viral',  'virus',  'visit',  'vital',  'vivid',  'vocal',  'voice',
    'voter',  'wagon',  'waist',  'waste',  'watch',  'water',  'wheat',  'wheel',  'where',  'which',  'while',  'white',
    'whole',  'whose',  'widow',  'width',  'winds',  'witch',  'woman',  'world',  'worry',  'worse',  'worst',  'worth',
    'would',  'wound',  'wrist',  'write',  'wrong',  'yield',  'young',  'yours',  'youth',  'zebra',
  ];

  return { WORDLE_WORDS: WORDS };
});
