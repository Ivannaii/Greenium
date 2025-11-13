from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import math

app = FastAPI(title="ESG Demo Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -------------------
# Data models (请求体)
# -------------------

class EnergyCostIn(BaseModel):
    monthly_energy_kwh: float = Field(..., gt=0, description="月均用电量 kWh")
    price_per_kwh: float = Field(..., gt=0, description="电价 元/kWh")
    area_m2: Optional[float] = Field(None, gt=0, description="建筑面积 m^2，非必须")
    roof_load_kg_per_m2: Optional[float] = Field(None, description="屋顶承重 kg/m^2，非必须")

class PVStorageIn(BaseModel):
    capacity_kw: float = Field(..., gt=0, description="光伏装机容量 kW")
    derate: float = Field(0.78, description="系统效率折减系数（组件+逆变器等） 0-1")
    annual_full_load_hours: float = Field(1200, description="年等效满发小时数")
    capex_per_kw: float = Field(5000, gt=0, description="每kW成本 元/kW")
    opex_per_year: float = Field(2000, description="年运维成本 元/年")
    self_consumption_ratio: float = Field(0.4, ge=0, le=1, description="自耗率（发电被自用比例）")
    feed_in_tariff: float = Field(0.3, description="上网电价 元/kWh")
    discount_rate: float = Field(0.08, description="折现率用于 NPV")
    lifetime_years: int = Field(20, description="项目寿命 年")

class StorageSimIn(BaseModel):
    capacity_kwh: float = Field(..., gt=0, description="储能容量 kWh")
    power_kw: float = Field(..., gt=0, description="额定充放电功率 kW")
    roundtrip_efficiency: float = Field(0.9, gt=0, le=1, description="一次往返效率")
    cycles_per_year: int = Field(365, description="年循环次数")
    degradation_per_year: float = Field(0.02, description="年容量衰减比例")

class CarbonIn(BaseModel):
    annual_grid_energy_kwh: float = Field(..., gt=0, description="年从电网购电量 kWh")
    grid_emission_factor: float = Field(0.6, description="电网排放因子 kgCO2e/kWh")
    pv_self_consumed_kwh: Optional[float] = Field(0.0, description="光伏自用量 kWh/年，作为减排")
    carbon_price_per_tco2: Optional[float] = Field(50.0, description="碳价 元/吨CO2e")

class ESGIn(BaseModel):
    energy_cost_yuan: float = Field(..., ge=0)
    annual_emission_tco2: float = Field(..., ge=0)
    payback_years: float = Field(..., ge=0)
    optional_indicators: Optional[dict] = None

# -------------------
# 算法实现函数
# -------------------

# 1) 能源成本与结构分析
def calc_energy_cost(monthly_energy_kwh: float, price_per_kwh: float, area_m2: Optional[float], roof_load: Optional[float]):
    annual_kwh = monthly_energy_kwh * 12.0
    annual_cost = annual_kwh * price_per_kwh
    intensity_kwh_per_m2 = (annual_kwh / area_m2) if area_m2 and area_m2 > 0 else None

    roof_advice = None
    if roof_load is not None:
        # 简单阈值：30 kg/m2 以上认为适合安装常规光伏组件与支架
        roof_advice = "适合安装" if roof_load >= 30 else "承重不足，需工程评估"

    return {
        "monthly_energy_kwh": monthly_energy_kwh,
        "annual_energy_kwh": round(annual_kwh, 2),
        "annual_cost_yuan": round(annual_cost, 2),
        "intensity_kwh_per_m2": round(intensity_kwh_per_m2,2) if intensity_kwh_per_m2 else None,
        "roof_advice": roof_advice
    }

# 2) 简化光伏发电与财务（逐年现金流、NPV、IRR 简化计算）
def estimate_pv_financials(capacity_kw: float, derate: float, annual_full_load_hours: float,
                           capex_per_kw: float, opex_per_year: float,
                           self_consumption_ratio: float, feed_in_tariff: float,
                           discount_rate: float, lifetime_years: int):
    # 年发电量（kWh）
    annual_gen = capacity_kw * annual_full_load_hours * derate
    # 年自用量与上网量
    self_used = annual_gen * self_consumption_ratio
    feed_in = annual_gen * (1 - self_consumption_ratio)
    # 年收益（自用按等效电价计算，此处近似用 feed_in_tariff 作为参考或传入具体电价）
    # 为简化，假设自用收益 = self_used * feed_in_tariff（或可换成更高的边际电价）
    annual_revenue = self_used * feed_in_tariff + feed_in * feed_in_tariff
    # 成本
    initial_capex = capacity_kw * capex_per_kw
    # 逐年现金流（按简化模型：收入-opex），不考虑税收与融资成本
    cashflows = []
    for y in range(1, lifetime_years + 1):
        # 假设系统每年衰减 0.5%（组件衰减）
        decay = (1 - 0.005) ** (y - 1)
        gen_y = annual_gen * decay
        revenue_y = gen_y * feed_in_tariff  # 简化模型
        cashflow_y = revenue_y - opex_per_year
        cashflows.append(round(cashflow_y, 2))
    # NPV 计算
    npv = -initial_capex
    for i, cf in enumerate(cashflows):
        npv += cf / ((1 + discount_rate) ** (i + 1))
    # 近似IRR：用二分法搜索
    def irr_from_cashflows(initial, flows):
        low, high = -0.99, 1.0
        for _ in range(60):
            mid = (low + high) / 2
            val = -initial
            for i, f in enumerate(flows):
                val += f / ((1 + mid) ** (i + 1))
            if val > 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2
    irr = irr_from_cashflows(initial_capex, cashflows)
    # Payback（简单累计现金流，不考虑折现）
    cum = -initial_capex
    payback = None
    for i, cf in enumerate(cashflows):
        cum += cf
        if cum >= 0:
            # 线性插值计算更精确的回本年
            prev_cum = cum - cf
            frac = (0 - prev_cum) / cf if cf != 0 else 0
            payback = round(i + frac, 2)  # year count (i starts from 0)
            break

    return {
        "annual_generation_kwh": round(annual_gen,2),
        "self_used_kwh": round(self_used,2),
        "feed_in_kwh": round(feed_in,2),
        "initial_capex_yuan": round(initial_capex,2),
        "annual_revenue_yuan": round(annual_revenue,2),
        "cashflows_yuan": cashflows,
        "npv_yuan": round(npv,2),
        "irr": round(irr,4),
        "payback_years": payback
    }

# 3) 储能简单仿真（按年，基于容量/功率/循环）
def simulate_storage_annually(capacity_kwh: float, power_kw: float, roundtrip_eff: float,
                              cycles_per_year: int, degradation_per_year: float):
    # 简化：假设每个循环放电量为 min(capacity, power*duration), 这里假设平均每次放电使用 0.8*capacity
    avg_dispatch_per_cycle = 0.8 * capacity_kwh
    annual_dispatched = avg_dispatch_per_cycle * cycles_per_year * (1 - degradation_per_year)
    # 有效输出考虑效率
    annual_output_kwh = annual_dispatched * roundtrip_eff
    # 简要性能指标
    return {
        "annual_dispatched_kwh": round(annual_dispatched,2),
        "annual_output_kwh": round(annual_output_kwh,2),
        "usable_capacity_kwh_start": round(capacity_kwh,2),
        "estimated_capacity_after_1y_kwh": round(capacity_kwh * (1 - degradation_per_year),2)
    }

# 4) 碳排放与碳积分
def carbon_accounting(annual_grid_kwh: float, grid_factor: float, pv_self_consumed_kwh: float, carbon_price_per_t: float):
    # annual emissions from grid
    emissions_kg = annual_grid_kwh * grid_factor
    emissions_t = emissions_kg / 1000.0
    # pv offset (self consumed)
    offset_kg = pv_self_consumed_kwh * grid_factor
    offset_t = offset_kg / 1000.0
    net_reduction_t = offset_t
    potential_credit_value = net_reduction_t * carbon_price_per_t
    return {
        "annual_emissions_tco2": round(emissions_t,3),
        "pv_offset_tco2": round(offset_t,3),
        "net_emissions_tco2": round(max(0, emissions_t - offset_t),3),
        "potential_carbon_credit_value_yuan": round(potential_credit_value,2)
    }

# 5) ESG评分与融资建议（规则引擎式）
def esg_scoring_and_advice(energy_cost_yuan: float, annual_emission_tco2: float, payback_years: float, optional_indicators: dict = None):
    # 简易打分规则（0-100）
    # E: 能源成本与排放共同决定。假设基准：成本越低、排放越低得分越高。
    e_score = max(0, 100 - (energy_cost_yuan / 10000.0) * 50 - (annual_emission_tco2 / 100.0) * 50)
    # S: 若没有具体数据，给中等分
    s_score = 70 + (optional_indicators.get("social") if optional_indicators and "social" in optional_indicators else 0) * 0.1
    # G: 若回本小于5年得分高
    g_score = 60 + max(0, (5 - payback_years) * 8) if payback_years > 0 else 60

    # clamp
    e_score = round(min(max(e_score, 0), 100),1)
    s_score = round(min(max(s_score, 0), 100),1)
    g_score = round(min(max(g_score, 0), 100),1)
    total = round((e_score + s_score + g_score) / 3.0,1)

    # Financing advice rules
    advice = []
    if payback_years and payback_years <= 5:
        advice.append("项目回收期较短，适合商业贷款或企业自筹。")
    else:
        advice.append("回收期较长，建议考虑 PPA、分期融资或政府补贴。")
    if annual_emission_tco2 > 50:
        advice.append("排放偏高，优先推进节能改造以提升ESG得分。")
    if total > 75:
        advice.append("ESG评分良好，可申请绿色债券/绿色贷款或碳融资。")

    return {
        "E": e_score,
        "S": s_score,
        "G": g_score,
        "total": total,
        "advice": advice
    }

# -------------------
# API endpoints
# -------------------

@app.post("/api/energy_cost")
def api_energy_cost(payload: EnergyCostIn):
    return calc_energy_cost(payload.monthly_energy_kwh, payload.price_per_kwh, payload.area_m2, payload.roof_load_kg_per_m2)

@app.post("/api/pv_roi")
def api_pv_roi(payload: PVStorageIn):
    # return PV financials and also basic storage-free estimation
    pv_result = estimate_pv_financials(
        capacity_kw=payload.capacity_kw,
        derate=payload.derate,
        annual_full_load_hours=payload.annual_full_load_hours,
        capex_per_kw=payload.capex_per_kw,
        opex_per_year=payload.opex_per_year,
        self_consumption_ratio=payload.self_consumption_ratio,
        feed_in_tariff=payload.feed_in_tariff,
        discount_rate=payload.discount_rate,
        lifetime_years=payload.lifetime_years
    )
    return {"pv": pv_result}

@app.post("/api/storage_simulate")
def api_storage_simulate(payload: StorageSimIn):
    return simulate_storage_annually(
        capacity_kwh=payload.capacity_kwh,
        power_kw=payload.power_kw,
        roundtrip_eff=payload.roundtrip_efficiency,
        cycles_per_year=payload.cycles_per_year,
        degradation_per_year=payload.degradation_per_year
    )

@app.post("/api/carbon_account")
def api_carbon_account(payload: CarbonIn):
    return carbon_accounting(
        annual_grid_kwh=payload.annual_grid_energy_kwh,
        grid_factor=payload.grid_emission_factor,
        pv_self_consumed_kwh=payload.pv_self_consumed_kwh or 0.0,
        carbon_price_per_t=payload.carbon_price_per_tco2
    )

@app.post("/api/esg_score")
def api_esg_score(payload: ESGIn):
    return esg_scoring_and_advice(
        energy_cost_yuan=payload.energy_cost_yuan,
        annual_emission_tco2=payload.annual_emission_tco2,
        payback_years=payload.payback_years,
        optional_indicators=payload.optional_indicators or {}
    )

# Simple root
@app.get("/")
def root():
    return {"message": "ESG Demo Backend alive"}