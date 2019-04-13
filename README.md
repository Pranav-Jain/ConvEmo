# ConvEmo;

#### EmoContext: [SEMEVAL’19](https://www.humanizing-ai.com/emocontext.html)


## Problem Statement

We want to analyse the sentiment of an utterance of
a user, eg:- a tweet, a reply, etc., leveraging the
lexical, semantic and contextual information of the
preceding conversation. Also, we’d like to know what
caused that sentiment, some trigger words, the
sentiment of a previous utterance, etc. This is an
ongoing challenge for SEMEVAL’19.

## Motivation

**_“Understanding Emotions in Textual Conversations is
a hard problem in absence of voice modulations and
facial expressions.”​_** Most of the solutions analysing
sentiment treat the utterances as individual entities
w/o leveraging the context of the whole
conversation, which can completely change the
meaning of an utterance. Also in this age of
micro-blogging, the strict bound on word limit makes
it a much more challenging task.

## Learning Task

Given a textual dialogue i.e. a user utterance along
with previous two utterances of the conversation, the
task is to classify the emotion of the final user
utterance as one of the emotion classes: Happy,
Sad, Angry or Others. Also, learn the class w/o the
previous conversation (For ​ **PLT​** ).

## Dataset

We’re using the dataset provided in the ​challenge​.
