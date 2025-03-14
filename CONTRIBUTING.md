# Welcome to the ProLIF contributing guide

Thank you for considering investing your time in contributing to our open-source project! There are
many ways you can contribute:

- Opening a new issue when you find a **bug**,
- Answering questions on the [Discussions pages](https://github.com/chemosim-lab/ProLIF/discussions),
- Improving the [documentation](https://prolif.readthedocs.io): this can be as little as fixing
  typos and broken links, to more involved work such as writing tutorials and blog posts,
- Submitting a pull-request: whether it is for fixing a bug, improving an existing
  functionality, or adding a new one.

In this guide you will get an overview of the contribution workflow from opening an issue, creating
a PR, reviewing, and merging the PR.


## New contributor guide

To get an overview of the project, read the [README](./README.md) file. Here are some general
resources to help you get started with open-source contributions:

- [Set up Git](https://docs.github.com/en/get-started/git-basics/set-up-git)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

And here are some resources that are specific to ProLIF:

- [Paper](https://doi.org/10.1186/s13321-021-00548-6): gives a good scientific overview of the
  library. Some implementation details and parameters are outdated.
- [Tutorials](https://prolif.readthedocs.io/en/stable/source/tutorials.html): great resource to
  understand how to use the package.
- [MDAnalysis UGM 2023 slides](https://github.com/MDAnalysis/UGM2023/blob/main/presentation_materials/bouysset_cedric/slides.pdf):
  contains an introduction to fingerprints (FPs) and the different flavors of interaction FPs, some
  examples of applications, and a general overview of the package.
- [RDKit UGM 2022 poster](https://drive.google.com/file/d/1F6-IUSKSfx2QFRCRqm0uBdWk7SQSSLKN/view):
  general overview of the library.
- [RDKitConverter Benchmark](https://github.com/MDAnalysis/RDKitConverter-benchmark): repo
  containing the benchmarks on the MDAnalysis-to-RDKit converter used by ProLIF when analysing
  MD trajectories and using the `prolif.Molecule.from_mda` interface.
- [RDKit UGM 2020 slides](https://github.com/rdkit/UGM_2020/blob/master/Presentations/C%C3%A9dricBouysset_From_RDKit_to_the_Universe.pdf):
  older slide deck detailing how the above converter works.
  

## Getting started

### Issues

#### Create a new issue

If you spot a problem, [search if an issue already exists](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments).

If a related issue doesn't exist, you can open a new issue using the relevant
[issue form](https://github.com/chemosim-lab/ProLIF/issues/new/choose). Try to be as detailed as
possible:

- Explain what you want to do and what would be the expected behavior of the code,
- Provide the full error message,
- Provide the code snippet that is causing the issue (as text, not as screenshots),
- Tell us what workarounds you have tried, if any.

#### Solve an issue

Scan through our [existing issues](https://github.com/chemosim-lab/ProLIF/issues) to find one that
interests you. You can narrow down the search using `labels` as filters. If you find an issue to
work on, you are welcome to open a PR with a fix.

### Make Changes

1. Fork the repository.

[Fork the repo](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository) 
so that you can make your changes without affecting the original project until you're ready to merge
them.

2. Setup a development environment.

We use [uv](https://docs.astral.sh/uv/) to manage our development environment. You don't have to use
it to make changes to the repo, this is only a recommendation, but it will make contributing to
ProLIF much easier.

- Follow their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
- Navigate to the forked ProLIF repository, and create a virtual environment:
  ```
  uv sync --python 3.11
  ```

3. Create a working branch and start with your changes!

Don't make your changes on the default `master` branch as it may make things harder to manage for
us. Instead, create a new branch and make your changes there.

Remember to add tests for any code changes (whether it's a fix or a new feature), and to write
an informative documentation (both in docstrings and tutorials).

4. Verify that your changes pass tests and other requirements.

We use [poethepoet](https://poethepoet.natn.io/) to define tasks to run in development environments.
You can run all of the checks detailed below using the command
```
uv run poe check
```

You can also get a list of available checks with:
```
uv run poe --help
```

  a. Running tests

To run the test suite, simply execute:
```
uv run poe test
```

  b. Building the documentation

Building the HTML files for the documentation and tutorials can be done with the following command:
```
uv run poe docs
```
You can then open the `docs/_build/html/index.html` file with your browser to navigate the docs and
see any changes that you've made.

If you're adding a new module, you will need to update some `.rst` files in the `docs/source/`
folder and/or the `modules` subfolder. For example, if you're adding a new functionality to plot
the IFP analysis results, you can make the relevant changes in the
`docs/source/modules/plotting.rst` file.

You will find the tutorials notebooks in the `docs/notebooks/` section. These are Jupyter-notebook
files with a twist for Markdown cells: you can use the
[MyST syntax](https://myst-nb.readthedocs.io/en/latest/authoring/jupyter-notebooks.html#syntax)
to format the content. See the [Authoring section](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html)
for more details.

  c. Code formatting and linting

You can check if your code complies with our code style standards with the following command:
```
uv run poe style-check
```

You can automatically format your changes to match with the style used in this project, as well as
fixing any lint errors (unused imports, type annotations...etc.) with the following command:

```
uv run poe style-fix
```


### Commit your update

Commit the changes once you are happy with them. Don't worry too much about your commit messages and
having a clear history as the changes will be squashed into a single commit before being merged into
the `master` branch. 

### Pull Request

When you're finished with the changes, create a pull request, also known as a PR.

- Use a descriptive title.
- In the description, start by
  [linking your PR to the issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue),
  e.g. `Fixes #123`.
- Fill in any details that will help us understand your changes. For example, if you tried different
  implementations and only the current iteration works, add it so that we don't ask you later
  *"Have you tried doing X instead?"*.
- Enable the checkbox to
  [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a
  merge.

Once you submit your PR, a maintainer will review your proposal. There's no need to assign a
reviewer or mention any maintainer in the PR.

Every time you make an update to your branch (e.g. a new commit is pushed), some automated checks
are run to ensure your code is compliant. If it's your first time contributing to ProLIF, a
maintainer will have to manually enable those automated checks. Once the results of these checks
are available, try to tackle any issue that was found.

We may ask questions or request additional information.

- We may ask for changes to be made before a PR can be merged, either using
[suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request)
or pull request comments. You can apply suggested changes directly through the UI. You can make any
other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as
  [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this
  [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge
  conflicts and other issues.

### Your PR is merged!

Congratulations :tada::tada: The ProLIF team thanks you :sparkles:.

Once your PR is merged, depending on other ongoing work, we may delay things a bit before making
a release with your changes.
