import DensityModelling_defs as dm

import apogee.tools.read as apread



allStar_trimmed = apread.allStar(main=True, rmdups=True)
allStar_not_trimmed = apread.allStar()
