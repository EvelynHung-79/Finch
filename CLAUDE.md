# Project Instructions

## Language
Always respond in Traditional Chinese (繁體中文) or English only. Never use Japanese or any other language.

## Memory Management
When saving or updating memory files, ALWAYS write to BOTH locations:
1. `/root/.claude/projects/-root-Finch/memory/` (Claude's default memory path)
2. `/root/Finch/.claude/memory/` (project-local copy, committed to git)

Both locations must stay in sync. The project-local copy at `.claude/memory/` is the source of truth that survives across GPU rentals.

## Content Management
Always read the content of the project files in `/root/Finch/doc/` before responding to any questions about the project. This directory contains all the documentation about the project, including research process, results, and details about the methods used.
