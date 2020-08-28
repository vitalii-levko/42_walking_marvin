#!/bin/bash
which -s brew
if [[ $? != 0 ]] ; then
	curl -fsSL https://rawgit.com/kube/42homebrew/master/install.sh | zsh
else
	brew update
fi
brew install swig
brew install ffmpeg
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip uninstall -y gym
pip uninstall -y pyglet
pip install -r requirements.txt
rm -rf env/lib/python3.7/site-packages/gym/envs
cp -R envs env/lib/python3.7/site-packages/gym
defaults write org.python.python ApplePersistenceIgnoreState NO
