sudo dnf install snapd
sudo snap install photoscape
sudo snap install gimp
sudo snap install mailspring
git clone https://github.com/flazz/vim-colorschemes ~/.vim/
plasmashell 5.20.4



os.environ["OMP_NUM_THREADS"] = "10"         # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "10"    # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "10"         # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "10"     # export NUMEXPR_NUM_THREADS=1

#--- Vim Commentary-------#
mkdir -p ~/.vim/pack/tpope/start
cd ~/.vim/pack/tpope/start
git clone https://tpope.io/vim/commentary.git
vim -u NONE -c "helptags commentary/doc" -c q
#_-----------------------------#
