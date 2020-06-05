# The Adapter-Hub Website

This repository builds our website, [adapterhub.ml](https://adapterhub.ml).
It's powered by Flask, [Frozen-Flask](https://pythonhosted.org/Frozen-Flask), Bootstrap and GitHub Pages.

**All content of the exploration pages is pulled from the [Hub repo](https://github.com/adapter-hub/hub). All content contributions should be made there.**

## Build 🛠

1. Clone this repository and update data from Hub:
```
git clone https://github.com/adapter-hub/website
git submodule init
git submodule update --remote
```
(The second and third command pull the latest data from the Hub repo.)

2. Install required Python libraries:
```
pip install -r requirements.txt
```

3. Install Bootstrap and required npm modules:
```
cd app/static && npm install
```
(Note: Building the website styles additionally requires [sass](https://sass-lang.com/) installed.)

4. Run 🚀
```
flask db init
flask run
```

5. Freeze ❄️ (optional). Freezing generates static pages that can be deployed to GitHub Pages.
```
flask freeze build
```
