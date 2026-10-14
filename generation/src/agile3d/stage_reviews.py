"""Stage-specific critics following AGILE v4 Appendix H, with local score checks.

Weights follow the paper. Numeric acceptance thresholds and finite retry budgets
are configurable implementation choices. Visible product print is valid; added
viewpoint captions are not. The original references remain the final authority.
"""

from typing import Annotated, Literal
from pydantic import Field
from .models import StrictModel

Score = Annotated[float, Field(ge=0, le=10)]
View = Literal["front", "back", "left", "right"]


class MultiviewReview(StrictModel):
    evidence_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["accept", "retry", "reject"]
    geometry_and_views: Score
    texture_and_material: Score
    geometric_details: Score
    feature_consistency: Score
    image_quality: Score
    has_view_labels: bool
    rotated_views: list[View] = Field(max_length=4)
    missing_views: list[View] = Field(max_length=4)
    view_positions: list[Literal["front", "back", "left", "right", "unknown"]] = Field(
        min_length=4, max_length=4
    )
    layout_penalty: float = Field(ge=0, le=2)
    matches_reference: bool
    no_hand_artifacts: bool
    feedback: str = Field(min_length=1, max_length=6000)

    def score(self):
        if self.has_view_labels:
            return 0.0
        value = (
            0.30 * self.geometry_and_views
            + 0.20 * self.texture_and_material
            + 0.20 * self.geometric_details
            + 0.15 * self.feature_consistency
            + 0.15 * self.image_quality
        )
        return round(
            max(0.0, value - 3 * len(set(self.rotated_views)) - self.layout_penalty), 6
        )

    def eligible(self, threshold):
        return (
            self.score() >= threshold
            and not self.has_view_labels
            and not self.rotated_views
            and not self.missing_views
            and self.view_positions == ["front", "back", "left", "right"]
            and self.matches_reference
            and self.no_hand_artifacts
        )


class TextureCorrection(StrictModel):
    severity: Literal["minor", "major", "severe"]
    region: str = Field(min_length=1, max_length=500)
    problem: str = Field(min_length=1, max_length=1000)
    edit_instruction: str = Field(min_length=1, max_length=1000)


class TextureReview(StrictModel):
    evidence_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["accept", "retry", "reject"]
    completeness: Score
    accuracy_no_fiction: Score
    correspondence: Score
    color_material: Score
    detail_fidelity: Score
    extra_penalty: float = Field(ge=0, le=10)
    invalid_texture: bool
    key_features_missing: bool
    severe_fiction: bool
    corrections: list[TextureCorrection] = Field(max_length=24)
    feedback: str = Field(min_length=1, max_length=6000)

    def score(self):
        if self.invalid_texture or self.key_features_missing or self.severe_fiction:
            return 0.0
        return round(
            max(
                0.0,
                0.30 * self.completeness
                + 0.25 * self.accuracy_no_fiction
                + 0.20 * self.correspondence
                + 0.15 * self.color_material
                + 0.10 * self.detail_fidelity
                - self.extra_penalty,
            ),
            6,
        )

    def eligible(self, threshold):
        return (
            self.score() >= threshold
            and not self.invalid_texture
            and not self.key_features_missing
            and not self.severe_fiction
            and not any(c.severity == "severe" for c in self.corrections)
        )
