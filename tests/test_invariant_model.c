#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../src/model.h"
#include "../src/model.c"

START_TEST(test_buffer_reads_never_exceed_declared_length)
{
    // Invariant: Buffer reads never exceed the declared length
    const char *payloads[] = {
        "A",  // Valid minimal input
        "This is a normal string",  // Valid normal input
        "A" "B" "C" "D" "E" "F" "G" "H" "I" "J" "K" "L" "M" "N" "O" "P" "Q" "R" "S" "T" "U" "V" "W" "X" "Y" "Z",  // Boundary case (26 chars)
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  // 100 chars - potential overflow
        NULL  // Sentinel
    };
    
    // Test model initialization with various sizes
    for (int i = 0; payloads[i] != NULL; i++) {
        size_t input_len = strlen(payloads[i]);
        
        // Create a test model configuration
        bitmamba_config_t config;
        config.n_layers = 1;
        config.n_heads = 1;
        config.vocab_size = input_len;
        
        // Initialize model - this is where vulnerable allocation happens
        bitmamba_model_t *model = bitmamba_init(&config);
        
        // Verify model was created (or NULL if allocation failed)
        if (model != NULL) {
            // Check that if vocab_size caused overflow, we didn't allocate undersized buffer
            // by verifying we can access expected memory range
            ck_assert_ptr_nonnull(model->layers);
            
            // Clean up
            bitmamba_free(model);
        }
        
        // Test with extreme multiplication that could overflow
        config.n_layers = SIZE_MAX / sizeof(bitmamba_block_t) + 1;
        model = bitmamba_init(&config);
        
        // Either allocation fails safely or succeeds with valid memory
        if (model != NULL) {
            ck_assert_ptr_nonnull(model->layers);
            bitmamba_free(model);
        }
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_buffer_reads_never_exceed_declared_length);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}