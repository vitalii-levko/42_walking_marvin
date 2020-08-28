# walking marvin
## prerequisites
python 3.7.6
## getting started
### install 42 homebrew
`curl -fsSL https://rawgit.com/kube/42homebrew/master/install.sh | zsh`
### install swig dependency for box2d
`brew install swig`
### install ffmpeg dependency for video rendering
`brew install ffmpeg`
### clone git repository
`git clone ...`
### create and activate virtual environment
`cd 42_marvin`\
`python3 -m venv env`\
`source env/bin/activate`
### upgrade `pip` and install required packages after uninstalling `gym` and `pyglet` if necessary
`python -m pip install --upgrade pip`\
`pip uninstall -y gym`\
`pip uninstall -y pyglet`\
`pip install -r requirements.txt`
### substitute your gym/envs folder with one attached to the project
`rm -rf env/lib/python3.7/site-packages/gym/envs`\
`cp -R envs env/lib/python3.7/site-packages/gym`
### suppress macOS warning
`defaults write org.python.python ApplePersistenceIgnoreState NO`
### run walking marvin
`python marvin.py`

