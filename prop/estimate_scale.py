
import datetime
from enum import Enum
import re

from regex import F
from sympy import Q


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
            sale: bool = None,
    ):
        self.datePoint = datePoint
        self.propType = propType
        self.prov = prov
        self.area = area
        self.city = city
        self.sale = sale

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
        for key in ['prov', 'area', 'city']:
            if self.__dict__[key]:
                keys.append(self.__dict__[key])
        if self.sale:
            keys.append('S')
        elif self.sale is False:
            keys.append('R')
        else:
            keys.append('-')
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
        return query

    def get_saletp_query(self):
        query = {}
        if self.sale is True:
            query['saletp'] = 'Sale'
        elif self.sale is False:
            query['saletp'] = 'Rent'
        return query

    def get_query(self):
        query = {'ptype': 'r'}
        query.update(self.get_type_query())
        query.update(self.get_geo_query())
        query.update(self.get_saletp_query())
        return query
