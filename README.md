# SABR Log-Normal for FX modelling (sabrfx)

This repository contains the code for FX modelling using SABR framework

## Motivation

SABR is a popular volatility model used to model volatility dynamics across multiple assets. Whilst literature on SABR Model is abundant, the implementation of the model applicable to the FX world is far from obvious. This includes:

* Premium adjusted Delta vs Non-Premium adjusted Delta
* "ATM" volatility quoted as delta-neutral straddle rather than at-the-money forward
* Market quotes the broker fly (or one-vol fly) rather than the smile fly

A robust framework would help users in the FX world to navigate round the conventions and arrive with numbers understandable by a model which take strike, volatility pairing and push out parameters which can be used to extrapolate/interpolate 

## Using the code

