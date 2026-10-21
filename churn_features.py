import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUM = ['age','tenure_months','monthly_charges','total_charges','complaints_last_12m','late_payments_last_12m','satisfaction_score','monthly_usage_gb','discount_pct','senior_citizen']
CAT = ['contract_type','internet_service','region','payment_method','paperless_billing','dependents','support_plan','tech_support']
ENG = ['charge_per_tenure','complaint_intensity','late_payment_intensity','dissatisfaction_index','service_support_gap','usage_to_charge_ratio','dissatisfaction_complaint_interaction']

class CustomerFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, engineered=True): self.engineered=engineered
    def fit(self,X,y=None):
        self.feature_names_in_=np.array(NUM+CAT,dtype=object)
        return self
    def transform(self,X):
        z=X[NUM+CAT].copy()
        if self.engineered:
            t=z.tenure_months.clip(lower=1); w=t.clip(upper=12)
            z['tenure_band']=pd.cut(t,[0,12,24,48,np.inf],labels=['0-12','13-24','25-48','49+']).astype(str)
            z['charge_per_tenure']=z.monthly_charges/t
            z['complaint_intensity']=z.complaints_last_12m/w
            z['late_payment_intensity']=z.late_payments_last_12m/w
            z['dissatisfaction_index']=10-z.satisfaction_score
            z['service_support_gap']=((z.internet_service!='No') & ((z.support_plan=='No') | (z.tech_support=='No'))).astype(int)
            z['usage_to_charge_ratio']=z.monthly_usage_gb/z.monthly_charges.where(z.monthly_charges>0)
            z['dissatisfaction_complaint_interaction']=z.dissatisfaction_index*z.complaints_last_12m
        return z.replace([np.inf,-np.inf],np.nan)
