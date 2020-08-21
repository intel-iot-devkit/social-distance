"""
Copyright (C) 2020 Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
"""

from jsonschema import validate as json_validate

schema = {"type": "object",
          "required": [
              "coords",
              "video",
              "pedestrian_model_weights",
              "pedestrian_model_description",
              "reidentification_model_weights",
              "reidentification_model_description"
          ],
          "additionalProperties": False,
          "properties": {
              "coords": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "number"

                            },
                            "minItems": 2,
                            "maxItems": 2
                        },
                  "minItems": 4,
                  "maxItems": 4
              },
              "video": {"type": "string", "maxLength": 150},
              "pedestrian_model_weights": {"type": "string", "maxLength": 250},
              "pedestrian_model_description": {"type": "string", "maxLength": 250},
              "reidentification_model_weights": {"type": "string", "maxLength": 250},
              "reidentification_model_description": {"type": "string", "maxLength": 250}
          }
}


def validate(datadict):
    return json_validate(instance=datadict, schema=schema)