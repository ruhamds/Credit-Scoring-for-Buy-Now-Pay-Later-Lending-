from pydantic import BaseModel, Field, PositiveInt, condecimal


class PredictionRequest(BaseModel):
    customer_id: PositiveInt
    credit_amount: condecimal(gt=0)
    term_months: PositiveInt
    annual_income: condecimal(gt=0)
    # add additional risk feature fields here


class PredictionResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    approved: bool
