document.addEventListener('DOMContentLoaded', function() {
    const ruleTypeSelect = document.getElementById('id_rule_type');
    const parametersJsonTextarea = document.getElementById('id_parameters_json');
    const parametersHelp = document.querySelector('span.help-block');

    function updateParametersHelp() {
        const ruleType = ruleTypeSelect.value;
        const helpTextMap = JSON.parse(parametersJsonTextarea.dataset.ruleTypeHelp);
        
        if (helpTextMap[ruleType]) {
            parametersHelp.textContent = helpTextMap[ruleType];
        } else {
            parametersHelp.textContent = 'Enter parameters as a valid JSON object.';
        }
    }

    // Initial setup
    updateParametersHelp();

    // Add event listener for rule type changes
    ruleTypeSelect.addEventListener('change', updateParametersHelp);
}); 