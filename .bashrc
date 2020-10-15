# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/senthil/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/senthil/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/senthil/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/senthil/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
source /home/senthil/anaconda3/bin/activate
export QT_DEBUG_PLUGINS=1
export QT_PLUGIN_PATH=/home/senthil/anaconda3/envs/mne/lib/python3.8/site-packages/PyQt5/Qt/plugins
export LD_LIBRARY_PATH=/home/senthil/anaconda3/envs/mne/lib/python3.8/site-packages/PyQt5/Qt/plugins
alias spyder='/home/senthil/anaconda3/envs/mnev2/bin/spyder > /dev/null 2>&1 &'
alias spss='/home/senthil/IBM/SPSS/Statistics/26/bin/stats 2>&1 &'
alias gpu1='lspci -v -s 08:00.0'
alias gpu2='glxinfo | egrep -i "device|memory"'
export FREESURFER_HOME=/usr/local/freesurfer
#export SUBJECTS_DIR=$FREESURFER_HOME/subjects
export SUBJECTS_DIR=/home/senthil/caesar/camcan/cc700/freesurfer_output
source $FREESURFER_HOME/SetUpFreeSurfer.sh
alias slicer='/home/senthil/Slicer-4.10.2-linux-amd64/Slicer > /dev/null 2>&1 &'
alias freeview='/usr/local/freesurfer/bin/freeview > /dev/null 2>&1 &'
alias fsleyes='~/fsl/bin/fsleyes > /dev/null 2>&1 &'
conda activate mne
alias mem='ps -o pid,user,%mem,command ax | sort -b -k3 -r'
export ANTSPATH=/home/senthil/anaconda3/envs/mne/bin
