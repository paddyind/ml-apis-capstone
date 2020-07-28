# Python: Getting Started

A barebones Django app, which can easily be deployed to Heroku.

This application supports the [Getting Started with Python on Heroku](https://devcenter.heroku.com/articles/getting-started-with-python) article - check it out.

## Running Locally

Make sure you have Python 3.7 [installed locally](http://install.python-guide.org). To push to Heroku, you'll need to install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli), as well as [Postgres](https://devcenter.heroku.com/articles/heroku-postgresql#local-setup).

```sh
$ git clone https://github.com/paddyind/ml-apis-capstone
$ cd ml-apis-capstone

$ python3 -m venv getting-started
$ pip install -r requirements.txt

$ heroku local
```
Your app should now be running on [localhost:9052](http://localhost:9052/).

## Deploying to Heroku

link git repository https://github.com/paddyind/ml-apis-capstone and deploy in Heroku Apps

## Documentation

For more information about using Python on Heroku, see these Dev Center articles:

- [Python on Heroku](https://devcenter.heroku.com/categories/python)