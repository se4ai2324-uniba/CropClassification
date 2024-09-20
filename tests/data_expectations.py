import datetime
import pandas as pd
import great_expectations as ge
import dvc.api


def create_expectation_suite():
    # Create a new Great Expectations context
    context = ge.get_context()
    datasource = context.datasources["crop_data_source"]
    asset = datasource.assets[0]
    batch_requests = asset.build_batch_request()

    context.add_or_update_expectation_suite("crop_recommendation_suite")

    validator = context.get_validator(
        batch_request=batch_requests,
        expectation_suite_name="crop_recommendation_suite"
    )

    validator.expect_column_to_exist("Nitrogen")
    validator.expect_column_to_exist("Phosphorus")
    validator.expect_column_to_exist("Potassium")
    validator.expect_column_to_exist("Temperature")
    validator.expect_column_to_exist("Humidity")
    validator.expect_column_to_exist("pH_Value")
    validator.expect_column_to_exist("Rainfall")
    validator.expect_column_to_exist("Crop")

    validator.expect_column_values_to_be_between("Nitrogen", min_value=0, max_value=200)
    validator.expect_column_values_to_be_between("Phosphorus", min_value=0, max_value=200)
    validator.expect_column_values_to_be_between("Potassium", min_value=0, max_value=200)
    validator.expect_column_values_to_be_between("Temperature", min_value=0, max_value=50)
    validator.expect_column_values_to_be_between("Humidity", min_value=0, max_value=100)
    validator.expect_column_values_to_be_between("pH_Value", min_value=0, max_value=14)
    validator.expect_column_values_to_be_between("Rainfall", min_value=0, max_value=300)

    validator.expect_column_values_to_not_be_null("Nitrogen")
    validator.expect_column_values_to_not_be_null("Phosphorus")
    validator.expect_column_values_to_not_be_null("Potassium")
    validator.expect_column_values_to_not_be_null("Temperature")
    validator.expect_column_values_to_not_be_null("Humidity")
    validator.expect_column_values_to_not_be_null("pH_Value")
    validator.expect_column_values_to_not_be_null("Rainfall")
    validator.expect_column_values_to_not_be_null("Crop")

    validator.save_expectation_suite(discard_failed_expectations=False)
    checkpoints = context.add_or_update_checkpoint(name="crop_recommendation_checkpoint", batch_request=batch_requests,expectation_suite_name="crop_recommendation_suite")
    checkpoint_result = checkpoints.run()
    context.view_validation_result(checkpoint_result)

    print(context.build_data_docs())


if __name__ == "__main__":
    create_expectation_suite()