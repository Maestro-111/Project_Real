
import datetime
from enum import Enum
import re

from regex import F


class PropertyType:
    DETACHED = 'Detached'
    SEMI_DETACHED = 'Semi-Detached'
    TOWNHOUSE = 'Townhouse'
    CONDO = 'Condo'
    OTHER = 'Other'


class PropertyTypeRegexp:
    DETACHED = re.compile(
        '^Det|Detached|Cottage|Det Condo|Det W/Com Elements|Link|Rural Resid', re.IGNORECASE)
    SEMI_DETACHED = re.compile(
        '^Semi-|Semi-Detached|Semi-Det Condo', re.IGNORECASE)
    TOWNHOUSE = re.compile(
        '^Att|Townhouse|Townhouse Condo|Att/Row/Twnhouse', re.IGNORECASE)
    CONDO = re.compile(
        'Apartment|Condo Apt|Co-Op Apt|Co-Ownership Apt|Comm Element Condo|Leasehold Condo', re.IGNORECASE)


class EstimateScale:
    """Scale of Estimation """

    def __init__(
            self,
            datePoint: datetime,
            propType: str = None,
            prov: str = 'ON',
            area: str = None,
            city: str = None,
            cmty: str = None):
        self.datePoint = datePoint
        self.propType = propType
        self.prov = prov
        self.area = area
        self.city = city
        self.cmty = cmty

    def __str__(self):
        keys = [self.datePoint.strftime('%Y-%m-%d')]
        if self.propType is None:
            keys.append('N')
        elif self.propType is PropertyType.DETACHED:
            keys.append('D')
        elif self.propType is PropertyType.SEMI_DETACHED:
            keys.append('S')
        elif self.propType is PropertyType.TOWNHOUSE:
            keys.append('T')
        elif self.propType is PropertyType.CONDO:
            keys.append('C')
        else:
            keys.append('O')
        for key in ['prov', 'area', 'city', 'cmty']:
            if self.__dict__[key]:
                keys.append(self.__dict__[key])
        return '.'.join(keys)

    def get_type_query(self):
        """Get the query for property type"""
        if self.propType is None:
            query = {}
        elif self.propType is PropertyType.DETACHED:
            query = {'ptype2': PropertyTypeRegexp.DETACHED}
        elif self.propType is PropertyType.SEMI_DETACHED:
            query = {'ptype2': PropertyTypeRegexp.SEMI_DETACHED}
        elif self.propType is PropertyType.TOWNHOUSE:
            query = {'ptype2': PropertyTypeRegexp.TOWNHOUSE}
        elif self.propType is PropertyType.CONDO:
            query = {'ptype2': PropertyTypeRegexp.CONDO}
        return query

    def get_geo_query(self):
        query = {}
        if self.prov:
            query['prov'] = self.prov
        if self.area:
            query['area'] = self.area
        if self.city:
            query['city'] = self.city
        if self.cmty:
            query['cmty'] = self.cmty
        return query
