

from base.util import allTypeToInt


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
    "Loht",
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
    "Apartment": 'bsmtApt',
    "Crw": 'bsmtCrw',
    "Fin": 'bsmtFin',
    "Finished": 'bsmtFin',
    "Full": 'bsmtFull',
    "Half": 'bsmtHalf',
    "NAN": 'bsmtNON',
    "None": 'bsmtNON',
    "NON": 'bsmtNON',
    "Prt": 'bsmtPrt',
    "Sep": 'bsmtSep',
    "Sep Entrance": 'bsmtSep',
    "Slab": 'bsmtSlab',
    "W/O": 'bsmtWO',
    "Finished/Walkout": ['bsmtFin', 'bsmtWO'],
    "Finish/Walkout": ['bsmtFin', 'bsmtWO'],
    "W/U": 'bsmtWU',
    "Y": 'bsmtY',
    "unFin": 'bsmtUnFin',
    "Unfinished": 'bsmtUnFin',
    "Other": 'bsmtOther',
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

acType = {
    '-': 'acNON',
    1.0: 'acY',
    4.0: 'acY',
    "?": 'acNON',
    "Cac": 'acCentral',
    "Central": 'acCentral',
    "Central Air": 'acCentral',
    "N": 'acNON',
    "None": 'acNON',
    "Other": 'acOther',
    "Part": 'acPart',
    "Wall Unit": 'acPart',
    "Window Unit": 'acPart',
    "Y": 'acY',
}


levelType = {
    1.0: 1,
    2.0: 2,
    3.0: 3,
    "2nd": 2,
    "2nd ": 2,
    "3rd": 3,
    "4Rd": 4,
    "4th": 4,
    "5th": 5,
    "Basement": 0,
    "Bsmnt": 0,
    "Bsmt": 0,
    "Flat": 1,
    "Ground": 1,
    "In Betwn": 1.5,
    "In-Betwn": 1.5,
    "Loft": 1,
    "Lower": 0.5,
    "M": 1,
    "Main": 1,
    "Sub-Bsmt": 0,
    "Upper": 1.5,
    # extra from rms
    "2Nd": 2,
    "3Rd": 3,
    "Above": 1.5,
    "Fihth": 5,
    "Fourth": 4,
    "Laundry": 1,
    "Other": 1,
    "Second": 2,
    "Sixth": 6,
    "Sub-Bsmt": 0,
    "Third": 3,
    "U": 1,
    "Unknown": 1,
}


def getLevel(label):
    if label in levelType:
        return levelType[label]
    l = allTypeToInt(label)
    if l is not None:
        return l
    return 1


garageType = {
    '-': 'GrNone',
    "None": 'GrNone',
    "Attached": 'GrAttached',
    "Boulevard": 'GrBoulevard',
    "Built-In": 'GrBuiltIn',
    "Built-in": 'GrBuiltIn',
    "Carport": 'GrCarport',
    "Covered": 'GrCovered',
    "Detached": 'GrDetached',
    "Double Detached": 'GrDoubleDetached',
    "In/Out": 'GrInOut',
    "Lane": 'GrLane',
    "Other": 'GrOther',
    "Ouside/Surface": 'GrSurface',
    "Outside/Surface": 'GrSurface',
    "Pay": 'GrPay',
    "Plaza": 'GrPlaza',
    "Public": 'GrPublic',
    "Reserved/Assignd": 'GrReserved',
    "Single Detached": 'GrSingleDetached',
    "Street": 'GrStreet',
    "Surface": 'GrSurface',
    "U": 'GrUnderground',
    "Undergrnd": 'GrUnderground',
    "Underground": 'GrUnderground',
    "Valet": 'GrValet',
    "Visitor": 'GrVisitor',
}

lockerType = {
    '-': 'lkrNone',
    'Common': 'lkrCommon',
    'Ensuite': 'lkrEnsuite',
    'Ensuite&Exclusive': 'lkrEnsuite',
    'Ensuite+Common': 'lkrEnsuite',
    'Ensuite+Exclusive': 'lkrEnsuite',
    'Ensuite+Owned': 'lkrEnsuite',
    'Exclusive': 'lkrExclusive',
    'None': 'lkrNone',
    'O': 'lkrOwned',
    'Owned': 'lkrOwned',
    1: 'lkrOwned',
    0: 'lkrNone',
}

exposureType = {
    '-': 'fceUnknown',
    'E': 'fceE',
    'N': 'fceN',
    'S': 'fceS',
    'W': 'fceW',
    'EW': ['fceE', 'fceW'],
    'Ew': ['fceE', 'fceW'],
    'NE': ['fceN', 'fceE'],
    'Ne': ['fceN', 'fceE'],
    'NS': ['fceN', 'fceS'],
    'Ns': ['fceN', 'fceS'],
    'NW': ['fceN', 'fceW'],
    'Nw': ['fceN', 'fceW'],
    'SE': ['fceS', 'fceE'],
    'Se': ['fceS', 'fceE'],
    'SW': ['fceS', 'fceW'],
    'Sw': ['fceS', 'fceW'],
}

heatType = {
    "_": "htOther",
    "Air circulation heat": 'htAir',
    "Baseboard": 'htBase',
    "Baseboard heaters": 'htBase',
    "Boiler": 'htBoiler',
    "Central Heat Pump": 'htCentral',
    "Central heating": 'htCentral',
    "Coil Fan": 'htCoil',
    "Ductless": 'htDuctless',
    "Elec Forced Air": ['htElec', 'htForcedAir'],
    "Elec Hot Water": ['htElec', 'htHotWater'],
    "Electric baseboard units": ['htElec', 'htBase'],
    "Fan Coil": 'htCoil',
    "Fao": 'htForcedAir',
    "Floor heat": 'htFloor',
    "Force Air": 'htForceAir',
    "Forced Air": 'htForceAir',
    "Forced Air Gas": ['htGas', 'htForceAir'],
    "Forced air": 'htForceAir',
    "Furnace": 'htFurnace',
    "Furnance": 'htFurnace',
    "Gas":  'htGas',
    "Gas Forced Air": ['htGas', 'htForceAir'],
    "Gas Forced Air Closd": ['htGas', 'htForceAir'],
    "Gas Forced Air Close": ['htGas', 'htForceAir'],
    "Gas Forced Air Open": ['htGas', 'htForceAir'],
    "Gas Hot Water": ['htGas', 'htHotWater'],
    "Gravity Heat System": 'htGravity',
    "Ground Source": 'htGround',
    "Ground Source Heat": 'htGround',
    "Heat Pump": 'htHeatPump',
    "Heat Recovery Ventilation (HRV)":  'htHRV',
    "High-Efficiency Furnace": 'htFurnace',
    "Hot Water": 'htHotWater',
    "Hot water radiator heat": 'htHotWater',
    "In Floor Heating": 'htFloor',
    "No heat": 'htNone',
    "None": 'htNone',
    "Not known": 'htNone',
    "Oil Forced Air": 'htOilForcedAir',
    "Oil Hot Water": 'htOilHotWater',
    "Oil Steam": 'htOilSteam',
    "Other": 'htOther',
    "Outside Furnace": 'htOutside',
    "Overhead Heaters": 'htOverhead',
    "Propane": 'htPropane',
    "Propane Gas": 'htPropane',
    "Radiant": 'htRadiant',
    "Radiant heat": 'htRadiant',
    "Radiant/Infra-red Heat": 'htRadiant',
    "Radiator": 'htRadiant',
    "See remarks": 'htOther',
    "Solar": 'htSolar',
    "Space Heater": 'htSpace',
    "Space heating baseboards": ['htSpace', 'htBase'],
    "Steam Radiators": 'htRadiant',
    "Steam radiator": 'htRadiant',
    "Stove": 'htStove',
    "Underfloor ducts": 'htDuctless',
    "Wall Heaters": 'htWall',
    "Wall Mounted Heat Pump": 'htWall',
    "Water": 'htWater',
    "Water Radiators": ['htWater', 'htRadiant'],
    "Wood Stove": 'htWood',
    "Woodburning":  'htWood',
}

fuelType = {
    "Bi energy": 'htBi',
    "Coal": 'htCoal',
    "Combination": 'htComb',
    "Electric": 'htElec',
    "Gas": 'htGas',
    "Gas ": 'htGas',
    "Geo Thermal": 'htGeo',
    "Grnd Srce": 'htGround',
    "Ground So": 'htGround',
    "Heat Pump": 'htHeatPump',
    "Heating oil": 'htOil',
    "Hot Water": 'htHotWater',
    "Natural gas": 'htGas',
    "Oil": 'htOil',
    "Other": 'htOther',
    "Pellet": 'htPellet',
    "Propane": 'htPropane',
    "See Remarks": 'htOther',
    "Solar": 'htSolar',
    "Unknown": 'htOther',
    "Waste oil": 'htOil',
    "Wind Power": 'htWind',
    "Wood": 'htWood',
}

laundryType = {
    'Coin Operated': 'lndryCoin',
    'Ensuite': 'lndryEnsuite',
    'In Area': 'lndryInArea',
    'Lower': 'lndryLower',
    'Main': 'lndryMain',
    'None': 'lndryNone',
    'Set Usage': 'lndrySetUsage',
    'Shared': 'lndryShared',
}

parkingDesignationType = {
    'Common': 'prkCommon',
    'Compact': 'prkCompact',
    'Exclusive': 'prkExclusive',
    'Mutual': 'prkMutual',
    'None': 'prkNone',
    'Owned': 'prkOwned',
    'Rental': 'prkRental',
    'Stacked': 'prkStacked',
    'U': 'prkU',
}

parkingFacilityType = {
    'F': 'prkFacility',
    'Facilities': 'prkFacility',
    'Mutual': 'prkMutual',
    'None': 'prkNone',
    'O': 'prkOther',
    'Other': 'prkOther',
    'Private': 'prkPrivate',
    'Surface': 'prkSurface',
    'U': 'prkUnderground',
    'Undergr': 'prkUnderground',
    'Undergrnd': 'prkUnderground',
}

balconyType = {
    "None": 0,
    "Open": 1,
    "Terr": 2,
    "Jlte": 3,
    "Encl": 4,
}


ptpType = {
    "Att/Row/Twnhouse",
    "Co-Op Apt",
    "Co-Ownership Apt",
    "Comm Element Condo",
    "Condo Apt",
    "Condo Townhouse",
    "Cottage",
    "Det Condo",
    "Det W/Com Elements",
    "Detached",
    "Duplex",
    "Fourplex",
    "Investment",
    "Leasehold Condo",
    "Link",
    "Mobile/Trailer",
    "Multiplex",
    "Room",
    "Rural Resid",
    "Semi-Det Condo",
    "Semi-Detached",
    "Shared Room",
    "Time Share",
    "Triplex",
    "Upper Level"
}

pstylType = {
    1.0: 1,
    2.0: 2,
    3.0: 3,
    "1 1/2 Storey": 1.5,
    "1 1/2-Storey": 1.5,
    "1-1/2 Storey": 1.5,
    "2 1/2 Storey": 1.5,
    "2 Storey": 2,
    "2-1/2 Storey": 2.5,
    "2-Storey": 2,
    "2.5 Storey": 2.5,
    "3 Storey": 3,
    "3- Storey": 3,
    "3-Storey": 3,
    "Apartment": 10,
    "Apts-13 To 20 Units": 13,
    "Apts-2 To 5 Units": 11,
    "Apts-6 To 12 Units": 12,
    "Apts-Over 20 Units": 14,
    "Bachelor/Studio": 1,
    "Backsplit 3": 5.4,
    "Backsplit 4": 5.4,
    "Backsplit 5": 5.5,
    "Backsplt-All": 5.6,
    "Bungaloft": 6,
    "Bungalow": 6.1,
    "Bungalow-Raised": 6.2,
    "Campgrounds": 7,
    "Industrial Loft": 7.1,
    "Loft": 7.2,
    "Multi-Level": 8,
    "Other": 0,
    "Professional Office": 9,
    "Residential": 0.5,
    "Retail": 0.1,
    "Retail Store Related": 0.1,
    "Seniors Residence": 0.6,
    "Service Related": 0.2,
    "Sidespilt 4": 6.4,
    "Sidesplit 3": 6.3,
    "Sidesplit 4": 6.4,
    "Sidesplit 4 ": 6.4,
    "Sidesplit 5": 6.5,
    "Sidesplt-All": 6.6,
    "Stacked Townhse": 5.3,
    "Warehouse Loft": 7.3,
}
