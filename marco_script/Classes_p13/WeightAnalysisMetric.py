from dataclasses import dataclass, asdict, field

@dataclass
class WeightAnalysisMetric():
    link_score: float
    is_link_score_line_max: bool
    is_sum_all_weight_superior_to_zero: bool

    def toJson(self) -> dict:
        return {'__WeightAnalysisMetric__': asdict(self)}

    @classmethod
    def fromJson(cls, data):
        return cls(**data)
