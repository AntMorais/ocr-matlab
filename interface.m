function interface()

    functionality = menu('Assignment 2', 'Train', 'Test');
    type = menu('Architecture', 'Associative Memory + Classifier', 'Classifier');
    act_func = menu('Activation Function', 'Purelin', 'Hardlim', 'Logsig');

    
    switch(functionality)
        case 1
            switch (type)
                case 1
                    switch(act_func)
                        case 1
                            assoc_mem_purelin();
                        case 2
                            assoc_mem_hardlim();
                        case 3
                            assoc_mem_logsig();
                    end
                case 2
                    switch(act_func)
                        case 1
                            class_purelin();
                        case 2
                            class_hardlim();
                        case 3
                            class_logsig();
                    end
            end
        case 2
            mpaper;
    end

return