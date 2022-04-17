# repro_bass

A repository for the Bass Connections 2021-2022 team on using AI to improve energy access, led by Kyle Bradbury.

# Tasks

[Reorganization tasks](https://docs.google.com/document/d/1nz_8KgZ67tHIYThxB4SnqBi43qTSxwanSBCCPWtvC1I/edit)

Tasks: 
- [ ] make a mark for all the scripts that people would explicitly call on and add documentation to each of them
- [ ] organize the /scratch/cek28 folder such that it looks like what jaden sent in student_announcements and set the directory for all the scripts to the /scratch/cek28
- [ ] create a "best practices" guide to how to run experiments. most likely somee video guides. This can be done by the end of the year.

# Organization of `/scratch/cek28`

*a copy of the message from Jaden Long sent in #student_announcements*
```
My group discussed a solid way to reorganize the /scratch/cek28 folder. The main reasoning is so that we can finally use GitHub to organize our scripts, and each team member can keep their own copy of the git repository that synchronizes through Github. Let me know how you like it:
folders under /scratch/cek28 (each user's git folder not shown here):
├── images - the folder for putting all the images and labels. Everyone will have read and write access to every part of it.
│   ├── BC_team_domain_experiments
│   ├── MW_images
│   └── ...
├── results - folder with all the results. Everyone will have read and write access to every part of it
│   └── ...
└── personal - the folder with everyone's folders where everyone puts what they want to share with others. Permissions for the folders under this directory will be under each user's discretion.
    ├── cek28
    ├── yl708
    └── ...
under each team member's home directory:
└── repro_bass - synchronized through GitHub
After we organize the folder like this, when we write scripts, we can all share the same file paths, and no one would have to make finicky changes for their own directory. This would permit us to share the same code through github. If there a user writes a script that requires to use its current directory, we can either achieve this through calling __file__ in Python, or creating another folder under that user's copy of the repro_bass folder that gets ignored by git through .gitignore.
```
