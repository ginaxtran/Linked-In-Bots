# Linked-In-Bots

## Overview:
Our project is to inform users of LinkedIn about the likelihood that a profile they are viewing could be generated by a bot, which Enhances transparency, Decreases the prevalence of false profiles
and Makes LinkedIn a more trustworthy platform for genuine users.

## Problem
* Overestimate the prevalence of bots 
* Exposure tends to impair judgment of bot-recognition self-efficacy 
* Increase propensity toward stricter bot-regulation policies 
* Decreased self-efficacy and increased perceptions of bot influence

## Research Question
What are characteristics or patterns we can use to detect LLM profile generated bots on Linkedin?

## Data Description
This dataset comprises a total of 3600 LinkedIn profiles, categorized into three main groups:

* Legitimate LinkedIn Profiles (LLPs): 1800 profiles
* Fake LinkedIn Profiles (FLPs): 600 profiles
* ChatGPT-generated LinkedIn Profiles (CLPs): 1200 profiles

## Model Labels

* Label 0: Legitimate LinkedIn Profiles (LLPs)
* Label 1: Fake LinkedIn Profiles (FLPs)
* Label 10: ChatGPT-generated LinkedIn Profiles (CLPs) created based on legitimate profiles' statistics
* Label 11: ChatGPT-generated LinkedIn Profiles (CLPs) created based on fake profiles' statistics

## Goals
We aim to inform users of LinkedIn about the likelihood that a profile they are viewing could be generated by a bot, which
Enhances transparency, Decreases the prevalence of false profiles, make LinkedIn a more trustworthy platform for genuine users

## Solution

We wanted to build a Chrome extension that could show the percentage that someone might be a bot if someone is viewing their profile. 
This would be determined by training an algorithm to look at keywords or behaviors of bots


### Citation
@inproceedings{ayoobi2023looming,
  title={The Looming Threat of Fake and LLM-generated LinkedIn Profiles: Challenges and Opportunities for Detection and Prevention},
  author={Ayoobi, Navid and Shahriar, Sadat and Mukherjee, Arjun},
  booktitle={Proceedings of the 34th ACM Conference on Hypertext and Social Media},
  pages={1--10},
  year={2023}
}
