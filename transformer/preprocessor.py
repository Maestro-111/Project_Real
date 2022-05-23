
import datetime
from itertools import chain

import pandas as pd
import re

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from base.base_cfg import BaseCfg
from prop.const import NONE, RENT_PRICE_UPPER_LIMIT, SALE_PRICE_LOWER_LIMIT, UNKNOWN, DROP, MEAN, Mode
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from prop.estimate_scale import PropertyType, PropertyTypeRegexp
from transformer.binary import BinaryTransformer
from transformer.one_hot_array import OneHotArrayEncodingTransformer
from transformer.select_col import SelectColumnTransformer
from transformer.db_label import DbLabelTransformer
from transformer.db_numeric import DbNumericTransformer
from transformer.drop_row import DropRowTransformer
from transformer.row_func import RowFuncTransformer
from transformer.simple_column import SimpleColumnTransformer

logger = BaseCfg.getLogger('preprocessor')


def yearOfDateNumber(dateNumber, deltaDays=0):
    date = datetime.datetime.strptime(str(dateNumber), '%Y%m%d')
    date = date + datetime.timedelta(days=deltaDays)
    return date.year


def yearOfByField(row, field, deltaDays=0):
    return yearOfDateNumber(row[field], deltaDays)


def yearOf(field, deltaDays=0):
    return lambda row: yearOfByField(row, field, deltaDays)


def saletpSingleValue(row, saletp):
    if isinstance(saletp, str):
        return saletp
    elif isinstance(saletp, tuple) or isinstance(saletp, list):
        if len(saletp) == 1:
            return saletp[0]
        elif len(saletp) == 0:
            return 'Sale'
        if row['lpr'] is None or row['lpr'] == 0:
            return 'Sale'
        elif row['lp'] is None or row['lp'] == 0:
            return 'Lease'
    return None


def binarySaletpByRow(row, saletp):
    value = saletpSingleValue(row, saletp)
    if value == 'Sale':
        return 0
    elif value == 'Lease':
        return 1
    elif value is not None:
        logger.error('Unknown saletp value', value, row['_id'])
    return None


def ptype2SingleValue(row, ptype2):
    for value in ptype2:
        if PropertyTypeRegexp.SEMI_DETACHED.match(value):
            return PropertyType.SEMI_DETACHED
        elif PropertyTypeRegexp.DETACHED.match(value):
            return PropertyType.DETACHED
        elif PropertyTypeRegexp.TOWNHOUSE.match(value):
            return PropertyType.TOWNHOUSE
        elif PropertyTypeRegexp.CONDO.match(value):
            return PropertyType.CONDO
    return None


def allTypeToInt(dummy, value):
    v_type = type(value)
    if v_type is int:
        return value
    if v_type is float:
        return int(value)
    if v_type is str:
        int_array = re.findall(r'\d+', "hello 42a I'm a m32 string 30")
        return int(''.join(int_array))
    if v_type is list:
        return allTypeToInt(None, value[0])
    return None


def shallDrop(row):
    try:
        if (row['saletp_b'] == 0):
            if (row['lp_n'] == 0) | (row['lp'] is None) | (row['lp'] < SALE_PRICE_LOWER_LIMIT):
                return True
        if (row['saletp_b'] == 1):
            if (row['lpr_n'] == 0) | (row['lpr'] is None) | (row['lpr'] > RENT_PRICE_UPPER_LIMIT):
                return True
        if (row['lst'] in ['Sld', 'Lsd']):
            if (row['sp_n'] == 0) | (row['sp'] is None):
                return True
    except Exception as e:
        logger.error(row)
        logger.error('shallDrop', e)
    return False


Level = {
    1.0: 1,
    2.0: 2,
    3.0: 3,
    "2nd": 2,
    "2nd ": 2,
    "3rd": 3,
    "4Rd": 4,
    "4th": 4,
    "5th": 5,
    "Basement": -1,
    "Bsmnt": -1,
    "Bsmt": -1,
    "Flat": 0,
    "Ground": 0,
    "In Betwn": 0.5,
    "In-Betwn": 0.5,
    "Loft": 1,
    "Lower": -0.5,
    "M": 0,
    "Main": 0,
    "Sub-Bsmt": -1,
    "Upper": 0.5,
    # extra from rms
    "2Nd": 2,
    "3Rd": 3,
    "Above": 0.5,
    "Fifth": 5,
    "Fourth": 4,
    "Laundry": 0,
    "Other": 0,
    "Second": 2,
    "Sixth": 6,
    "Sub-Bsmt": -1,
    "Third": 3,
    "U": 0,
    "Unknown": 0,
}


def getLevel(label: str):
    if label in Level:
        return Level[label]
    l = allTypeToInt(None, label)
    if l is not None:
        return l
    return 0


roomType = [
    "  ",
    "1 Pc Bath",
    "1pc Bathroom",
    "1pc Ensuite bath",
    "2 Bedroom",
    "2 Pc Bath",
    "2Br",
    "2nd Bedro",
    "2nd Br",
    "2nd Br ",
    "2pc Bathroom",
    "2pc Ensuite bath",
    "3 Bedroom",
    "3 Pc Bath",
    "3Br",
    "3pc Bathroom",
    "3pc Ensuite bath",
    "3rd Bedro",
    "3rd Br",
    "4 Bedroom",
    "4 Pc Bath",
    "4Br",
    "4pc Bathroom",
    "4pc Ensuite bath",
    "4th B/R",
    "4th Bedro",
    "4th Br",
    "5 Pc Bath",
    "5pc Bathroom",
    "5pc Ensuite bath",
    "5th Bedro",
    "5th Br",
    "6 Pc Bath",
    "6pc Bathroom",
    "6pc Ensuite bath",
    "6th Br",
    "7 Pc Bath",
    "7th Br",
    "Addition",
    "Additional bedroom",
    "Atrium",
    "Attic",
    "Attic (finished)",
    "B Liv Rm",
    "Balcony",
    "Bar",
    "Bath",
    "Bath (# pieces 1-6)",
    "Bathroom",
    "Bed",
    "Bedrom",
    "Bedroom",
    "Bedroom 2",
    "Bedroom 3",
    "Bedroom 4",
    "Bedroom 5",
    "Bedroom 6",
    "Beverage",
    "Bonus",
    "Bonus Rm",
    "Br",
    "Breakfast",
    "Breakfest",
    "Closet",
    "Cold",
    "Cold Rm",
    "Cold/Cant",
    "Coldroom",
    "Common Rm",
    "Common Ro",
    "Computer",
    "Conservatory",
    "Den",
    "Din",
    "Dinette",
    "Dining",
    "Dining Rm",
    "Dining nook",
    "Dinning",
    "Eat in kitchen",
    "Eating area",
    "Enclosed porch",
    "Ensuite",
    "Ensuite (# pieces 2-6)",
    "Entrance",
    "Exer",
    "Exercise",
    "Fam",
    "Fam Rm",
    "Family",
    "Family Rm",
    "Family bathroom",
    "Family/Fireplace",
    "Flat",
    "Flex Space",
    "Florida",
    "Florida/Fireplace",
    "Foyer",
    "Fruit",
    "Fruit cellar",
    "Full bathroom",
    "Full ensuite bathroom",
    "Furnace",
    "Game",
    "Games",
    "Great",
    "Great Rm",
    "Great Roo",
    "Guest suite",
    "Gym",
    "Hall",
    "Hobby",
    "Indoor Pool",
    "Inlaw suite",
    "Kit",
    "Kitchen",
    "Kitchen/Dining",
    "L Porch",
    "Laundry",
    "Laundry / Bath",
    "Library",
    "Living",
    "Living ",
    "Living Rm",
    "Living/Dining",
    "Living/Fireplace",
    "Lobby",
    "Locker",
    "Loft",
    "Master",
    "Master Bd",
    "Master bedroom",
    "Mbr",
    "Media",
    "Media/Ent",
    "Mezzanine",
    "Mud",
    "Mudroom",
    "Muskoka",
    "Nook",
    "Not known",
    "Nursery",
    "Office",
    "Other",
    "Pantry",
    "Partial bathroom",
    "Partial ensuite bathroom",
    "Patio",
    "Play",
    "Play Rm",
    "Playroom",
    "Porch",
    "Powder Rm",
    "Powder Ro",
    "Prim Bdrm",
    "Primary",
    "Primary B",
    "Primary Bedroom",
    "Rec",
    "Rec Rm",
    "Recreatio",
    "Recreation",
    "Recreational, Games",
    "Rental unit",
    "Roughed-In Bathroom",
    "Sauna",
    "Second Kitchen",
    "Sitting",
    "Solarium",
    "Steam",
    "Storage",
    "Studio",
    "Study",
    "Sun",
    "Sun Rm",
    "Sunroom",
    "Sunroom/Fireplace",
    "Tandem",
    "Tandem Rm",
    "U Porch",
    "Utility",
    "Walk Up Attic",
    "Wet Bar",
    "Wine Cellar",
    "Work",
    "Workshop"
]

bsmtType = {
    "Apt": 'bsmtApt',
    "Crw": 'bsmtCrw',
    "Fin": 'bsmtFin',
    "Full": 'bsmtFull',
    "Half": 'bsmtHalf',
    "NAN": 'bsmtNON',
    "NON": 'bsmtNON',
    "Prt": 'bsmtPrt',
    "Sep": 'bsmtSep',
    "Slab": 'bsmtSlab',
    "W/O": 'bsmtWO',
    "W/U": 'bsmtWU',
    "Y": 'bsmtY',
    "unFin": 'bsmtUnFin',
}

featType = {
    "Arts Centre": 'featArtsCentre',
    "Beach": 'featBeach',
    "Bush": 'featBush',
    "Campground": 'featCampground',
    "Clear View": 'featClearView',
    "Cul De Sac": 'featCulDeSac',
    "Cul De Sac/Deadend": 'featCulDeSac',
    "Cul Desac/Dead End": 'featCulDeSac',
    "Cul-De-Sac": 'featCulDeSac',
    "Dead End": 'featCulDeSac',
    "Electric Car Charg": 'featElectricCarCharg',
    "Electric Car Charger": 'featElectricCarCharg',
    "Equestrian": 'featEquestrian',
    "Fenced Yard": 'featFencedYard',
    "Garden Shed": 'featGardenShed',
    "Geenbelt/Conser.": 'featGeenbelt',
    "Golf": 'featGolf',
    "Greenbelt/Conse": 'featGreenbelt',
    "Greenbelt/Conserv": 'featGreenbelt',
    "Greenbelt/Conserv.": 'featGreenbelt',
    "Greenblt/Conser": 'featGreenbelt',
    "Grnbelt/Conserv": 'featGreenbelt',
    "Grnbelt/Conserv.": 'featGreenbelt',
    "Hospital": 'featHospital',
    "Island": 'featIsland',
    "Lake Access": 'featLakeAccess',
    "Lake Backlot": 'featLakeBacklot',
    "Lake Pond": 'featLakePond',
    "Lake/Pond": 'featLakePond',
    "Lake/Pond/River": 'featLakePond',
    "Lakefront/River": 'featLakePond',
    "Level": 'featLevel',
    "Library": 'featLibrary',
    "Major Highway": 'featMajorHighway',
    "Marina": 'featMarina',
    "Other": 'featOther',
    "Park": 'featPark',
    "Part Cleared": 'featPartCleared',
    "Part Cleared ": 'featPartCleared',
    "Place Of Workship": 'featPlaceOfWorkship',
    "Place Of Workshop": 'featPlaceOfWorkship',
    "Place Of Worship": 'featPlaceOfWorship',
    "Public": 'featPublic',
    "Public Transit": 'featPublicTransit',
    "Ravine": 'featRavine',
    "Rec Centre": 'featRecCentre',
    "Rec Rm": 'featRecRm',
    "Rec/Comm Centre": 'featRecCentre',
    "Rec/Commun Centre": 'featRecCentre',
    "Rec/Commun Ctr": 'featRecCentre',
    "Rec/Commun.Ctr": 'featRecCentre',
    "River/Stream": 'featRiverStream',
    "Rolling": 'featRolling',
    "School": 'featSchool',
    "School Bus Route": 'featSchoolBusRoute',
    "Security System": 'featSecuritySystem',
    "Skiing": 'featSkiing',
    "Sking": 'featSkiing',
    "Sloping": 'featSloping',
    "Slopping": 'featSloping',
    "Stucco/Plaster": 'featStuccoPlaster',
    "Terraced": 'featTerraced',
    "Tiled": 'featTiled',
    "Tiled/Drainage": 'featTiled',
    "Treed": 'featTreed',
    "Waterfront": 'featWaterfront',
    "Wood": 'featWood',
    "Wood/Treed": 'featWood',
    "Wooded/Treed": 'featWood',
}

constrType = {
    "A. Siding/Brick": 'ConstrBrick',
    "Alum": 'ConstrAlum',
    "Alum Siding": 'ConstrAlum',
    "Alum Slding": 'ConstrAlum',
    "Aluminium Siding": 'ConstrAlum',
    "Aluminum": 'ConstrAlum',
    "Aluminum Siding": 'ConstrAlum',
    "Aluminum Sliding": 'ConstrAlum',
    "Board/Batten": 'ConstrBoard',
    "Brick": 'ConstrBrick',
    "Brick Front": 'ConstrBrick',
    "Concrete": 'ConstrConc',
    "Insulbrick": 'ConstrInsul',
    "Log": 'ConstrLog',
    "Metal/Side": 'ConstrMetal',
    "Metal/Sliding": 'ConstrMetal',
    "Metal/Steel": 'ConstrMetal',
    "Other": 'ConstrOther',
    "Shingle": 'ConstrShing',
    "Stocco (Plaster)": 'ConstrStucco',
    "Stone": 'ConstrStone',
    "Stone(Plaster)": 'ConstrStone',
    "Stucco (Plaster)": 'ConstrStucco',
    "Stucco Plaster": 'ConstrStucco',
    "Stucco(Plaster)": 'ConstrStucco',
    "Stucco/Plaster": 'ConstrStucco',
    "Vinyl": 'ConstrVinyl',
    "Vinyl Siding": 'ConstrVinyl',
    "Vinyl Slding": 'ConstrVinyl',
    "Vinyl Sliding": 'ConstrVinyl',
    "Wood": 'ConstrWood',
}


class Preprocessor(TransformerMixin, BaseEstimator):
    """ Transforms raw training and prediction data

    Transforming steps:
    -. drop na columns if in training mode, return error if in prediction mode.
    -. convert binary saletp to 0 or 1, column name as 'saletp_b'
    -. convert ptype2 to single value. column name as 'ptype2_l'
    -. convert binary cols. column name as '_b'
    -. convert categorical columns to integers. column name as '_n'
    -. fill numeric columns to default values. column name as '_n'
    -. filter lat/lng to the range of [-180, 180] and drop null rows.
    """
    # binary use index as value, default 0
    cols_binary: dict = {
        'status':       ['U', 'A'],
        'den_fr':       ['N', 'Y'],
        'ens_lndry':    ['N', 'Y'],
        'cac_inc':      ['N', 'Y'],
        'comel_inc':    ['N', 'Y'],
        'heat_inc':     ['N', 'Y'],
        'prkg_inc':     ['N', 'Y'],
        'hydro_inc':    ['N', 'Y'],
        'water_inc':    ['N', 'Y'],
        'tv':           ['N', 'Y'],
        'all_inc':      ['N', 'Y'],
        'furnished':    ['N', 'Y'],
        'retirement':   ['N', 'Y'],
    }
    cols_label: dict = {  # na DROP means to remove the rows without label
        'lst':      {'na': UNKNOWN},
        'prov':     {'na': DROP},
        'area':     {'na': UNKNOWN},
        'city':     {'na': DROP},
        'cmty':     {'na': UNKNOWN},
        'st':       {'na': UNKNOWN},
        'zip':      {'na': UNKNOWN},
        'ptype':    {'na': 'r'},
        #        'saletp':   {'na': DROP},
        'ptype2_l': {'na': DROP},
        'pstyl':    {'na': UNKNOWN},
        'ptp':      {'na': UNKNOWN},
        'zone':     {'na': UNKNOWN},
        'gatp':     {'na': NONE},
        'heat':     {'na': 'Forced Air'},
        'fuel':     {'na': UNKNOWN},
        'balcony':  {'na': NONE},
        'laundry':  {'na': NONE},
        'laundry_lev': {'na': NONE},
        'fce':      {'na': 'U'},
        'lkr':      {'na': NONE},
        'rltr':     {'na': UNKNOWN},
        'pets':     {'na': UNKNOWN},
        'park_desig': {'na': UNKNOWN},
        'park_fac':  {'na': UNKNOWN},
    }
    cols_numeric: dict = {
        'lat':      {'na': DROP},
        'lng':      {'na': DROP},
        #        'st_num':   {'na': 0},
        'mfee':     {'na': 0},
        'tbdrms':   {'na': 0},
        'bdrms':    {'na': 0},
        'br_plus':  {'na': 0},
        'bthrms':   {'na': 0},
        'kch':      {'na': 0},
        'kch_plus': {'na': 0},
        'tgr':      {'na': 0},
        'gr':       {'na': 0},
        'lp':       {'na': 0},
        'lpr':      {'na': 0},
        'sp':       {'na': 0},
    }
    cols_special: dict = {
        'lp':       {'na': DROP},
        'lpr':      {'na': DROP},
        'sp':       {'na': DROP},
        # extract number parts from street number
        'st_num':   {'to': 'st_num_n'},
        'sqft':     {'to': 'sqft_n'},  # from sqft or rmSqft or sqft estimator
        'rmSqft':   {'to': 'sqft_n'},  # rmSqft or sqft estimator
        # from bltYr or rmBltYr or bltYr estimator
        'bltYr':    {'to': 'built_yr_n'},
        'rmBltYr':  {'to': 'built_yr_n'},  # rmBltYr or bltYr estimator
        'ptype2':   {'to': 'ptype2_l'},  # ptype2
    }
    cols_array_label: dict = {
        'constr':   constrType,
        'feat':     featType,
        'bsmt':     bsmtType,
    }

    cols_condo: list[str] = [
        'unt'  # get unit and level or penhouse. C5628585
    ]
    cols_structured: list[str] = [
        'rms',  # get primary bedroom dimensions, sum of all bedrooms deminsions
        'bths',  # get bath numbers on each level

        'la',  # la id : la.agnt[].id
        'la2',  # la2 id: la2.agnt[].id
        'schools',  # get 3 school names and ratings, rankings
    ]
    cols_forsale_in_models: dict = {
        'tax':      {'na': MEAN},
        # the first half year counted as previous tax year
        'taxyr':    {'na': lambda row: yearOfByField(row, 'onD', -183)},
    }
    cols_forsale_house_in_models: dict = {
        # MEAN of all depths when Detached/Semi-Detached/Freehold Townhouse
        'depth':    {'na': MEAN},
        # MEAN of all flt when Detached/Semi-Detached/Freehold Townhouse
        'flt':      {'na': MEAN},
    }

    def __init__(self, mode: Mode = Mode.PREDICT, collection_prefix: str = 'ml_'):
        self.mode: Mode = mode
        self.collection_prefix: str = collection_prefix
        self.label_collection = self.collection_prefix + 'label'
        self.number_collection = self.collection_prefix + 'number'

    def build_transformer(self, all_cols):
        """Build the transformers for baseline transforming.
           The transformers need to work with less columns when the predictions has less data.

            Parameters
            ----------
            all_cols : list of strings
                all columns to transform

        """
        logger.info('Building transformers for baseline transforming')
        all_cols = [*all_cols]

        colTransformerParams = [
            ('built_year', SelectColumnTransformer(
                col='built_yr_n', columns=['bltYr', 'rmBltYr'], v_type=int, as_na_value=None)),
            ('sqft', SelectColumnTransformer(
                col='sqft_n', columns=['sqft', 'rmSqft'], v_type=int, as_na_value=None)),
            ('st_num', allTypeToInt, 'st_num', 'st_num_n'),
            ('saletp_b', binarySaletpByRow, 'saletp', 'saletp_b'),
            ('ptype2_label', ptype2SingleValue, 'ptype2', 'ptype2_l'),
        ]
        all_cols.append('saletp_b')
        all_cols.append('ptype2_l')
        # array labels
        for k, v in self.cols_array_label.items():
            if k in all_cols:
                transformer = OneHotArrayEncodingTransformer(k, v, '_b')
                colTransformerParams.append((f'{k}_x', transformer))
                all_cols.extend(transformer.target_cols())
        # binary columns
        for k, v in self.cols_binary.items():
            if k in all_cols:
                colTransformerParams.append(
                    (f'{k}_b', BinaryTransformer(v, k), k, f'{k}_b'))
                all_cols.append(f'{k}_b')
        # categorical columns
        for k, v in self.cols_label.items():
            if k in all_cols:
                colTransformerParams.append(
                    (f'{k}_label', DbLabelTransformer(
                        self.label_collection,
                        col=k,
                        mode=self.mode,
                        na_value=v['na'],), k, f'{k}_n'))
                all_cols.append(f'{k}_n')
        # numeric columns
        for k, v in self.cols_numeric.items():
            if k in all_cols:
                colTransformerParams.append(
                    (f'{k}_number', DbNumericTransformer(
                        self.number_collection,
                        col=k,
                        mode=self.mode,
                        na_value=v['na'],), k, f'{k}_n'))
                all_cols.append(f'{k}_n')
        # drop rows: this operation has to be done outside of the pipeline
        # the drop operation is dependent on the model's usage of columns
        drop_na_cols = []
        for k, v in chain(self.cols_label.items(), self.cols_numeric.items()):
            if (k in all_cols) and (v['na'] is DROP):
                drop_na_cols.append(k)
        self.drop_na_cols_ = drop_na_cols
        colTransformerParams.append(
            ('drop_na', DropRowTransformer(drop_cols=drop_na_cols, mode=self.mode))
        )
        colTransformerParams.append(
            ('drop_check', DropRowTransformer(drop_func=shallDrop, mode=self.mode))
        )

        # create the pipeline
        self.customTransformer = SimpleColumnTransformer(
            colTransformerParams)

    def fit(self, Xdf: pd.DataFrame, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        self.build_transformer(Xdf.columns)
        self.customTransformer.fit(Xdf, y)
        self.n_features_ = Xdf.shape[1]
        return self

    def transform(self, Xdf: pd.DataFrame):
        """Transform the dataframe to the baseline format.

            Parameters
            ----------
            df : pd.DataFrame
                the dataframe to transform

        """
        if self.n_features_ is None:
            raise ValueError('The transformer has not been fitted yet.')

        if self.customTransformer is None:
            self.build_transformer(Xdf.columns)
            self.customTransformer.fit(Xdf)
        logger.info('Transforming to baseline format')
        #pd.set_option('mode.chained_assignment', None)
        Xdf = self.customTransformer.transform(Xdf)
        #pd.set_option('mode.chained_assignment', 'warn')
        return Xdf
