// Validation functionality
const ValidationModule = {
    // Initialize validation functionality
    init: function() {
        this.bindEvents();
        this.setupFormValidation();
        this.setupDynamicRules();
    },

    // Bind event listeners
    bindEvents: function() {
        // Rule type selection
        $('#ruleType').on('change', this.handleRuleTypeChange);
        
        // Dataset selection
        $('#dataset').on('change', this.handleDatasetChange);
        
        // Validation form submission
        $('#validationForm').on('submit', this.handleValidationSubmit);
    },

    // Handle rule type changes
    handleRuleTypeChange: function(e) {
        const ruleType = $(this).val();
        $('.rule-params').hide();
        $(`#${ruleType}Params`).show();
        
        // Update parameter requirements
        ValidationModule.updateParameterRequirements(ruleType);
    },

    // Handle dataset selection
    handleDatasetChange: function(e) {
        const datasetId = $(this).val();
        if (datasetId) {
            ValidationModule.loadDatasetColumns(datasetId);
        }
    },

    // Load dataset columns
    loadDatasetColumns: function(datasetId) {
        $.get(`/api/dataset/${datasetId}/columns`, function(columns) {
            const columnSelect = $('#columnName');
            columnSelect.empty();
            columns.forEach(column => {
                columnSelect.append(new Option(column, column));
            });
        });
    },

    // Setup form validation
    setupFormValidation: function() {
        // Bootstrap form validation
        const forms = document.querySelectorAll('.needs-validation');
        Array.from(forms).forEach(form => {
            form.addEventListener('submit', event => {
                if (!form.checkValidity()) {
                    event.preventDefault();
                    event.stopPropagation();
                }
                form.classList.add('was-validated');
            }, false);
        });
    },

    // Setup dynamic rule parameters
    setupDynamicRules: function() {
        // Initialize rule type parameters
        const initialRuleType = $('#ruleType').val();
        if (initialRuleType) {
            this.updateParameterRequirements(initialRuleType);
        }
    },

    // Update parameter requirements based on rule type
    updateParameterRequirements: function(ruleType) {
        const paramContainer = $(`#${ruleType}Params`);
        
        // Clear existing parameters
        paramContainer.empty();
        
        // Add parameters based on rule type
        switch(ruleType) {
            case 'range':
                paramContainer.append(`
                    <div class="mb-3">
                        <label for="minValue" class="form-label">Minimum Value</label>
                        <input type="number" class="form-control" id="minValue" name="parameters.min" required>
                    </div>
                    <div class="mb-3">
                        <label for="maxValue" class="form-label">Maximum Value</label>
                        <input type="number" class="form-control" id="maxValue" name="parameters.max" required>
                    </div>
                `);
                break;
                
            case 'format':
                paramContainer.append(`
                    <div class="mb-3">
                        <label for="pattern" class="form-label">Regex Pattern</label>
                        <input type="text" class="form-control" id="pattern" name="parameters.pattern" required>
                        <div class="form-text">Enter a valid regular expression pattern</div>
                    </div>
                `);
                break;
                
            case 'unique':
                paramContainer.append(`
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="caseSensitive" name="parameters.case_sensitive">
                            <label class="form-check-label" for="caseSensitive">Case Sensitive</label>
                        </div>
                    </div>
                `);
                break;
        }
    },

    // Handle validation form submission
    handleValidationSubmit: function(e) {
        e.preventDefault();
        
        const form = $(this);
        const url = form.attr('action');
        
        $.ajax({
            type: 'POST',
            url: url,
            data: form.serialize(),
            success: function(response) {
                ValidationModule.handleValidationSuccess(response);
            },
            error: function(xhr) {
                ValidationModule.handleValidationError(xhr);
            }
        });
    },

    // Handle successful validation
    handleValidationSuccess: function(response) {
        // Show success message
        const toast = new bootstrap.Toast($('#validationSuccessToast'));
        $('#validationSuccessMessage').text(response.message);
        toast.show();
        
        // Update validation results
        this.updateValidationResults(response.results);
    },

    // Handle validation error
    handleValidationError: function(xhr) {
        // Show error message
        const toast = new bootstrap.Toast($('#validationErrorToast'));
        $('#validationErrorMessage').text(xhr.responseJSON?.message || 'An error occurred during validation');
        toast.show();
    },

    // Update validation results display
    updateValidationResults: function(results) {
        const resultsContainer = $('#validationResults');
        resultsContainer.empty();
        
        if (results.length === 0) {
            resultsContainer.append('<div class="alert alert-success">All validations passed!</div>');
            return;
        }
        
        const table = $('<table class="table table-striped">').appendTo(resultsContainer);
        table.append(`
            <thead>
                <tr>
                    <th>Row</th>
                    <th>Column</th>
                    <th>Value</th>
                    <th>Error</th>
                </tr>
            </thead>
        `);
        
        const tbody = $('<tbody>').appendTo(table);
        results.forEach(result => {
            tbody.append(`
                <tr>
                    <td>${result.row}</td>
                    <td>${result.column}</td>
                    <td>${result.value}</td>
                    <td>${result.error}</td>
                </tr>
            `);
        });
    }
};

// Initialize validation module when document is ready
$(document).ready(function() {
    ValidationModule.init();
}); 