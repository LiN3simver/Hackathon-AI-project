"""Seat-specific screening and shared visual grounding for both interiors."""

from app.services.model_time_limits import RecognitionModel, run_with_time_limit


class ModelUnavailableError(RuntimeError):
    pass


class InvalidModelResponseError(RuntimeError):
    pass


class RoutedCleanlinessAnalyzer:
    def analyze(self, content: bytes, expected_view: str | None = None):
        if expected_view == "front":
            from time import perf_counter

            from app.services.cleanliness_localization import apply_localization
            from app.services.front_cleanliness import get_front_cleanliness_analyzer
            started = perf_counter()
            assessment, _ = run_with_time_limit(
                RecognitionModel.FRONT_CLEANLINESS,
                get_front_cleanliness_analyzer().analyze, content, expected_view,
            )
            return apply_localization(content, expected_view, assessment), (perf_counter() - started) * 1000
        if expected_view == "rear":
            from time import perf_counter

            from app.services.cleanliness_localization import apply_localization
            from app.services.rear_cleanliness import get_rear_cleanliness_analyzer

            started = perf_counter()
            assessment, _ = run_with_time_limit(
                RecognitionModel.REAR_CLEANLINESS,
                get_rear_cleanliness_analyzer().analyze, content, expected_view,
            )
            return apply_localization(content, expected_view, assessment), (perf_counter() - started) * 1000
        raise InvalidModelResponseError("整潔辨識請指定前座或後座。")


_analyzer = RoutedCleanlinessAnalyzer()


def get_cleanliness_analyzer() -> RoutedCleanlinessAnalyzer:
    return _analyzer
