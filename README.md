![](https://github.com/dyahalomi/democratic_detrender/blob/main/logo.png)

Source code for the democratic detrender.

See the [documentation](https://democratic-detrender.readthedocs.io/en/latest/) for 
installation instructions, tutorials, and the API.

## Installation
To install, you can run:
pip install git+https://github.com/dyahalomi/democratic_detrender

If you want to use the problem time identifying module in Jupyter Notebook you'll want to:
pip install PyQt5

And then in your notebook call:
%matplotlib qt

For figure output, you'll want to switch back to:
%matplotlib inline


## Citation

If you use this code in your research, please cite Yahalomi, Kipping, et al. 2026 [ADS](https://ui.adsabs.harvard.edu/abs/2026ApJS..283...51Y/abstract) / [arXiv](https://arxiv.org/abs/2411.09753):

    @ARTICLE{2026ApJS..283...51Y,
       author = {{Yahalomi}, Daniel A. and {Kipping}, David and {Solano-Oropeza}, Diana and {Li}, Madison and {Poddar}, Avishi and {Zhang}, Xunhe (Andrew) and {Abaakil}, Yassine and {Cassese}, Ben and {Jennings}, Jeff and {Larsen}, Skylar and {Turner}, Jake D. and {Teachey}, Alex and {Liu}, Jiajing and {Sundai}, Farai and {Valaskovic}, Lila},
        title = "{The democratic detrender: Ensemble-based Removal of the Nuisance Signal in Stellar Time-series Photometry}",
      journal = {\apjs},
     keywords = {Open source software, Transits, Astronomy data analysis, Astrostatistics, Stellar activity, 1866, 1711, 1858, 1882, 1580, Earth and Planetary Astrophysics, Instrumentation and Methods for Astrophysics, Solar and Stellar Astrophysics},
         year = 2026,
        month = apr,
       volume = {283},
       number = {2},
          eid = {51},
        pages = {51},
          doi = {10.3847/1538-4365/ae43c2},
archivePrefix = {arXiv},
       eprint = {2411.09753},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026ApJS..283...51Y},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

