# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
export PATH=~/anaconda3/bin:$PATH
PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=
#export QT_DEBUG_PLUGINS=1
#export QT_PLUGIN_PATH=/home/senthilp/anaconda3/envs/mne/lib/python3.8/site-packages/PyQt5/Qt/plugins
#export LD_LIBRARY_PATH=/home/senthilp/anaconda3/envs/mne/lib/python3.8/site-packages/PyQt5/Qt/plugins
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
# alias fsleyes='~/fsl/bin/fsleyes > /dev/null 2>&1 &'
alias mem='ps -o pid,user,%mem,command ax | sort -b -k3 -r'
export ANTSPATH=/home/senthilp/anaconda3/envs/mne/bin
alias connect='cd /home/senthilp/anaconda3/envs/mne/lib/python3.8/site-packages/mne/connectivity'
alias matlab='/usr/local/bin/matlab 2>&1 &'
alias watch='watch -n 1 free -g'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/senthilp/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/senthilp/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/senthilp/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/senthilp/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Git branch in prompt version 2
parse_git_branch() {
git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

conda activate mne
alias cpu='lscpu | grep -e Socket -e Core -e Thread'
alias smi='nvidia-smi'
alias kernel='yum info kernel -q'
#export LD_LIBRARY_PATH=/home/senthilp/anaconda3/envs/mne/lib:$LD_LIBRARY_PATH
source /opt/rh/devtoolset-9/enable
alias data='cd /home/senthilp/caesar/camcan/cc700/freesurfer_output'
