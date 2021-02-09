# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs
export PATH=~/anaconda3/bin:$PATH
source /home/senthilp/anaconda3/bin/activate
#export QT_DEBUG_PLUGINS=1
#export QT_PLUGIN_PATH=/home/senthilp/anaconda3/envs/mne/lib/python3.8/site-packages/PyQt5/Qt/plugins
#export LD_LIBRARY_PATH=/home/senthilp/anaconda3/envs/mne/lib/python3.8/site-packages/PyQt5/Qt/plugins

# Git branch in prompt version 2
parse_git_branch() {
 git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}
export PS1="\u@\h \W\[\033[32m\]\$(parse_git_branch)\[\033[00m\] $ "
alias spyder='/home/senthilp/anaconda3/envs/mne/bin/spyder > /dev/null 2>&1 &'
alias spss='/home/senthilp/IBM/SPSS/Statistics/26/bin/stats 2>&1 &'
alias gpu1='lspci -v -s 08:00.0'
alias gpu2='glxinfo | egrep -i "device|memory"'
export FREESURFER_HOME=/home/senthilp/freesurfer
#export SUBJECTS_DIR=$FREESURFER_HOME/subjects
export SUBJECTS_DIR=/home/senthilp/caesar/camcan/cc700/freesurfer_output
source $FREESURFER_HOME/SetUpFreeSurfer.sh
alias slicer='/home/senthilp/Slicer-4.11.20200930-linux-amd64/Slicer > /dev/null 2>&1 &'
alias freeview='/home/senthilp/freesurfer/bin/freeview > /dev/null 2>&1 &'
alias mem='ps -o pid,user,%mem,command ax | sort -b -k3 -r'
# FSL Setup
#FSLDIR=/home/senthilp/fsl
#PATH=${FSLDIR}/bin:${PATH}
#export FSLDIR PATH
#. ${FSLDIR}/etc/fslconf/fsl.sh
export ANTSPATH=/home/senthilp/anaconda3/envs/mne/bin
# alias fsleyes='~/fsl/bin/fsleyes > /dev/null 2>&1 &'
alias connect='cd /home/senthilp/anaconda3/envs/mne/lib/python3.8/site-packages/mne/connectivity'
alias matlab='/usr/local/bin/matlab 2>&1 &'
alias watch='watch -n 1 free -g'
conda activate mne
alias cpu='lscpu | grep -e Socket -e Core -e Thread'
alias smi='nvidia-smi'
alias kernel='yum info kernel -q'
source /opt/rh/devtoolset-9/enable
alias data='cd /home/senthilp/caesar/camcan/cc700/freesurfer_output'
