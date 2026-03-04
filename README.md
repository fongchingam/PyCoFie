<img src="https://github.com/fongchingam/PyCoFie/blob/main/PyCoFie_logo.png" width="600" />

# PyCoFie
  * **Py**thon-based **Co**rona **Fie**ld modeling tool.
  * Authors: Chingam Fong, Kenny C. Y. Ng (the Chinese University of Hong Kong)
  * Can be used to analytically calculate Current Sheet Source Surface (CSSS) model and Potential Field Source Surface (PFSS) model on a preset grid, or at any arbitrary point.
  * Incorporating sunpy, with field line tracing and visualization funcationalities.
  * Multi-threading capable. Reasonably fast.
  
# Method
  * The functions follows the same logic as the IDL code by Prof. Xuepu Zhao, hosted on Stanford solar group website http://sun.stanford.edu/~xuepu/DATA/
  * Mathmatical formulism:
    * PFSS: Hoeksema, 1984; Wang and Sheeley Jr., 1992
    * CSSS: Zhao and Hoeksema, 1994; Zhao and Hoeksema, 1995  
    * Notes on PFSS-like models: Xudong Sun, http://wso.stanford.edu/words/pfss.pdf
  
# Acknowledgement  
  * We thank Prof. Xudong Sun, Dr. Guanglu Shi for useful discussions, Prof. J. Todd Hoeksema for permission to reproduce the code. 
  
## Change logs:
  * 2026-03-04 pre-alpha, all functions made public on GitHub, with 1 notebook showing basic usage
