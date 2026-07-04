---
title: "How an Engineer Budgets"
description: ""
tags: [ projects, software ]
date_effective: 2026-07-04
draft: true
---

## Motivation

I have been writing software since I was 13 years old. I am now almost 34. When I was an adolescent, my mother worked for a company that taught people how to invest in stocks, options, futures, forex, and more. Financial education was part of my childhood --- and naturally, one of my first software projects was budgeting software.

Around that time, budgeting software was transitioning from desktop applications to cloud-based services, and companies were clamoring to copy Mint's success. So many budgeting websites emerged, and I watched most of them wither into obscurity. Nearly all of them were very simple, and their idea of a budget was limited to categorizing expenses and reserving a fixed dollar amount each month for each category. None of them were a tool for planning for the future the way I had been taught.

I first wrote my budgeting software as a Windows Forms application in Visual Basic .NET and --- if I recall correctly --- stored the data as an XML file. A few years later, I rewrote it in C#, still using Windows Forms, but switched from XML to binary serialization for the save data. A few years after that, I lost the source code but continued using the software myself.

Then a friend asked me to help him build a budget in Excel. That led me to recreate my system in VBA. I used that version for a while, but Excel felt too rigid for what I needed. Around this time, I was also managing my own income as a college student, paid by the hour while juggling school and inconsistent paychecks. Managing an hourly income while balancing school made it increasingly apparent that a budgeting system built around predictable monthly income was a poor fit for many people. Life is messy, and I wanted budgeting software that reflected that reality rather than an idealized one.

Fast forward to 2019 --- I had a stroke during my final semester of college. I limped toward the finish line and graduated, but the next several years were dedicated to recovery and discovering new diagnoses that affected my life. I never established a stable career in that time. By 2025, when my situation started to improve dramatically, I began thinking about how to reintegrate myself into the job market.

I built this website and created a Discord bot, but --- as an electrical engineer --- JavaScript, CSS, and Python are not exactly the skills I want to showcase to a potential employer. They are skills I picked up while putting myself through college doing web development, but web development was never my passion. Those projects were valuable for shaking the dust off my programming skills, but I wanted something closer to home: a project at a slightly lower level using C++, a language I considered one of my strongest skills but hadn't touched in seven years. When I last used it, C++14 was the current standard.

FundOS was born at the intersection of those interests: a budgeting system I had rewritten a dozen times throughout my life that didn't exist anywhere else; a desire for local control over my data instead of relying on a cloud service or subscription; a way to ease myself back into the skill set I had let grow rusty over the years; a portfolio project to demonstrate my abilities; and an opportunity to reconnect with how C++ had evolved since I last worked with it.

## What Makes FundOS Different
