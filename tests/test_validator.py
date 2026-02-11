from src.gen_ai.validator import parse_numeric,validate_data_plausibility


def test_parse_numeric_basic():
    assert parse_numeric("40,000 km") == 40000.0

def test_parse_numeric_none():
    assert parse_numeric(None) is None

def test_parse_numeric_currency():
    assert parse_numeric("$25,500") == 25500.0


# Invalid year (too old)
def test_validate_data_plausibility_invalid_yr():
    warnings = validate_data_plausibility("toyota","camry",1900,34000,45000)
    print(warnings)
    assert "Year is implausible" in warnings


# Invalid future year
def test_validate_data_plausibility_future_year():
    warnings = validate_data_plausibility("toyota","camry",2035,20000,30000)
    print(warnings)
    assert "Year is implausible" in warnings


# Suspiciously low price for new car
def test_validate_data_plausibility_low_price_new_car():
    warnings = validate_data_plausibility("tesla","model 3",2023,10000,8000)
    print(warnings)
    assert any("Price Alert" in w for w in warnings)


# High km per year usage
def test_validate_data_plausibility_high_usage():
    warnings = validate_data_plausibility("mazda","cx5",2020,200000,22000)
    print(warnings)
    assert any("High Usage" in w for w in warnings)


# Suspiciously low kms
def test_validate_data_plausibility_low_kms():
    warnings = validate_data_plausibility("hyundai","tucson",2021,500,27000)
    print(warnings)
    assert any("Suspiciously Low Kms" in w for w in warnings)


# Multiple warnings triggered
def test_validate_data_plausibility_multiple_flags():
    warnings = validate_data_plausibility("tesla","model y",2023,80000,9000)
    print(warnings)
    assert len(warnings) >= 2


# Valid realistic case (no warnings expected)
def test_validate_data_plausibility_valid_case():
    warnings = validate_data_plausibility("toyota","camry",2019,60000,25000)
    print(warnings)
    assert warnings == []


# Zero kms edge case
def test_validate_data_plausibility_zero_kms():
    warnings = validate_data_plausibility("toyota","corolla",2022,0,22000)
    print(warnings)
    assert isinstance(warnings, list)


# Negative price edge case (if your logic allows parsing)
def test_validate_data_plausibility_negative_price():
    warnings = validate_data_plausibility("kia","sportage",2018,50000,-100)
    print(warnings)
    assert isinstance(warnings, list)


# Non-numeric input handled by parse_numeric
def test_validate_data_plausibility_non_numeric():
    warnings = validate_data_plausibility("toyota","camry","abcd","xyz","price")
    print(warnings)
    assert warnings == []


