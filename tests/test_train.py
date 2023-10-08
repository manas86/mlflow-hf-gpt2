import unittest
from unittest.mock import Mock, patch
import mlflow
from mlflow.pyfunc import PythonModel
from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train import generate_text, GPT2Model, main

class TestGenerateText(unittest.TestCase):
    def test_generate_text(self):
        prompt = "Once upon a time,"
        max_length = 50
        generated_text = generate_text(prompt, max_length)
        self.assertIsInstance(generated_text, str)

class TestGPT2Model(unittest.TestCase):
    @patch("train.GPT2LMHeadModel.from_pretrained")
    @patch("train.GPT2Tokenizer.from_pretrained")
    def test_load_context(self, mock_tokenizer, mock_model):
        # Mock the model creation
        mock_model.return_value = Mock()

        gpt2_model = GPT2Model()
        gpt2_model.load_context(None)
        # Verify that the model was created
        self.assertIsNotNone(gpt2_model.model)

    def test_predict(self):
        gpt2_model = GPT2Model()
        gpt2_model.model = Mock()
        gpt2_model.model.generate.return_value = [[1, 2, 3]]
        model_input = "Test input"
        result = gpt2_model.predict(None, model_input)
        self.assertIsInstance(result, str)

class TestMainFunction(unittest.TestCase):
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.register_model")
    @patch("mlflow.pyfunc.log_model")
    @patch("mlflow.end_run")
    def test_main(self, mock_end_run, mock_log_model, mock_register_model, mock_start_run, mock_set_experiment):
        main()
        self.assertTrue(mock_set_experiment.called)
        self.assertTrue(mock_start_run.called)
        self.assertTrue(mock_log_model.called)
        self.assertTrue(mock_register_model.called)
        self.assertTrue(mock_end_run.called)

if __name__ == '__main__':
    unittest.main()
