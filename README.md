# VegSWBM

## Get the repo
`cd <path/to/where/you/want/the/repo>`
`git clone https://github.com/JohannaTrost/VegSWBM.git`
`cd VegSWBM`

## Tracking changes

1. `git status` - prints your changes and all necessary information.
2. `git add .` - will add all changes ("." means the current directory).
3. `git add <file>` or `git add <folder>` - adds only a specific file or folder.
4. `git status` - will now tell you that you have uncomitted changes or so.
5. `git commit <file> -m "Put a useful message about your changes"` - "commiting" is an essential step to track all changes and their history (<file> can also be a folder and if you don't specify it all the things you "added" will be commited). Things that have not been added (with `git add`) cannot be commited!
6. `git status` - if all you have committed all changes then `git status` will tell you.

Now, your changes are well tracked! But only on you local computer! You need to push them to the remote repo to make them available for everyone:

7. `git push`

